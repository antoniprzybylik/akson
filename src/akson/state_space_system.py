import math
import torch
import torch.nn as nn
import mini_ode
import numpy as np
import sympy as sp
from ._tf2ss import tf2ss
from ._utils import resolve_default_solver
import warnings
from typing import Tuple, Optional, Callable


class LinearF(nn.Module):
    """! Linear state equation module: dx/dt = A x + B (u - u0)

    The operating point input u0 defaults to zero, in which case
    the equation is dx/dt = A x + B u.
    """

    def __init__(
        self, A: torch.Tensor, B: torch.Tensor, u0: Optional[torch.Tensor] = None
    ):
        """! Construct LinearF
        @param A The dynamics matrix
        @param B The input matrix
        @param u0 The operating point input
        """
        super().__init__()
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        if u0 is None:
            u0 = torch.zeros(B.shape[1], dtype=A.dtype, device=A.device)
        self.register_buffer("u0", u0)

    def forward(
        self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """! Evaluate LinearF
        @param t The current system time
        @param x The current state
        @param u The current system input

        @return Derivative of the state x
        """
        du = u - self.u0
        return self.A @ x + self.B @ du


class LinearG(nn.Module):
    """! Linear output equation module: y = y0 + C x + D (u - u0)

    u0/y0 default to zero, in which case the equation is
    y = C x + D u.
    """

    def __init__(
        self,
        C: torch.Tensor,
        D: torch.Tensor,
        u0: Optional[torch.Tensor] = None,
        y0: Optional[torch.Tensor] = None,
    ):
        """! Construct LinearG
        @param A The dynamics matrix
        @param B The input matrix
        @param u0 The operating point input
        @param y0 The operating point output
        """
        super().__init__()
        self.register_buffer("C", C)
        self.register_buffer("D", D)
        if u0 is None:
            u0 = torch.zeros(D.shape[1], dtype=C.dtype, device=C.device)
        if y0 is None:
            y0 = torch.zeros(D.shape[0], dtype=C.dtype, device=C.device)
        self.register_buffer("u0", u0)
        self.register_buffer("y0", y0)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """! Evaluate LinearG
        @param x The current state
        @param u The current system input

        @return System output
        """
        du = u - self.u0
        return self.y0 + self.C @ x + self.D @ du


class OperatingPoint:
    """! An independent (u, x, y) triple describing an equilibrium point
    used as a reference for step-response analysis or as a default
    simulation starting state.
    """

    def __init__(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """! Bare-bones constructor: only checks u/x/y are 1D tensors and
        puts them on a common dtype/device.

        Calling this constructor directly is dangerous because it
        performs NO validation that (u, x, y) is an equilibrium
        point and does NOT check the model constraints.

        Prefer one of the validated constructors instead:
          - `from_values`: constructs an operating point from provided
            values while checking equilibrium and constraint conditions.
          - `from_input`: Finds an operating point for a given input. It
            fails if such an operating point does not exist.
        """
        if u.ndim != 1:
            raise ValueError(f"u must be 1D. Got shape {u.shape}")
        if x.ndim != 1:
            raise ValueError(f"x must be 1D. Got shape {x.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D. Got shape {y.shape}")

        self.dtype = dtype or u.dtype
        self.device = device or u.device
        self.u = u.clone().to(dtype=self.dtype, device=self.device)
        self.x = x.clone().to(dtype=self.dtype, device=self.device)
        self.y = y.clone().to(dtype=self.dtype, device=self.device)

    def __repr__(self) -> str:
        return f"OperatingPoint(u={self.u.tolist()}, x={self.x.tolist()}, y={self.y.tolist()})"

    @staticmethod
    def from_values(
        u: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        dynamics: Optional["StateSpaceDynamics"] = None,
        tol: float = 1e-8,
    ) -> "OperatingPoint":
        """! Construct an OperatingPoint from explicit values, optionally
        validating it against a model.

        @param dynamics If given, verifies that (u, x) is (approximately) an
            equilibrium of `dynamics` - i.e. ||f(0, x, u)|| <= tol - raising
            ValueError otherwise. Also cross-checks the supplied `y` against
            g(x, u) (warns, doesn't raise, on mismatch - y is sometimes
            deliberately a rounded/nominal literature value that doesn't
            exactly reproduce from the identified model), and validates
            u/x/y against `dynamics`'s constraints.
            If None, no model-based validation is performed at all.
        @param tol Absolute tolerance for the equilibrium/consistency checks.
        """
        if dynamics is not None:
            if u.shape != (dynamics.n_inputs,):
                raise ValueError(
                    f"u has bad shape {u.shape}. Expected ({dynamics.n_inputs},)"
                )
            if x.shape != (dynamics.state_size,):
                raise ValueError(
                    f"x has bad shape {x.shape}. Expected ({dynamics.state_size},)"
                )
            if y.shape != (dynamics.n_outputs,):
                raise ValueError(
                    f"y has bad shape {y.shape}. Expected ({dynamics.n_outputs},)"
                )

        op = OperatingPoint(
            u,
            x,
            y,
            dtype=dynamics.dtype if dynamics is not None else None,
            device=dynamics.device if dynamics is not None else None,
        )

        if dynamics is not None:
            t0_tensor = torch.as_tensor(0.0, dtype=op.dtype, device=op.device)
            with torch.no_grad():
                residual = dynamics.f_original(t0_tensor, op.x, op.u)
            residual_norm = float(residual.norm())
            if residual_norm > tol:
                raise ValueError(
                    "(u, x) is not an equilibrium of the given dynamics: "
                    f"||f(0, x, u)|| = {residual_norm:.3e} exceeds tol={tol:.1e}. "
                    "If you don't know x exactly, consider OperatingPoint.from_input(dynamics, u) "
                    "to find it numerically instead."
                )
            with torch.no_grad():
                y_check = dynamics.g_original(op.x, op.u)
            y_mismatch = float((y_check - op.y).norm())
            if y_mismatch > tol:
                warnings.warn(
                    f"Supplied y={op.y.tolist()} does not match g(x, u)={y_check.tolist()} "
                    "computed from the model at this (x, u).",
                    RuntimeWarning,
                )
            dynamics._validate_u(op.u)
            dynamics._validate_x(op.x)
            dynamics._validate_y(op.y)

        return op

    @staticmethod
    def from_input(
        dynamics: "StateSpaceDynamics",
        u: torch.Tensor,
        x_init: Optional[torch.Tensor] = None,
        tol: float = 1e-8,
        initial_duration: float = 1.0,
        max_attempts: int = 20,
    ) -> "OperatingPoint":
        """! Numerically find a stable equilibrium of `dynamics` for a
        constant input `u`, by simulating for increasingly long
        durations - doubling the duration each attempt - until
        ||f(0, x, u)|| <= tol.

        This is a HEURISTIC search, not a proof of stability: it can be
        fooled by e.g. an extremely slow transient that looks converged
        within `max_attempts`.

        @param dynamics The dynamics model.
        @param u Constant input to find the equilibrium for.
        @param x_init Starting state for the search.
        @param tol Convergence tolerance on ||f(0, x, u)||.
        @param initial_duration First simulation duration to try; doubled on
            each subsequent attempt.
        @param max_attempts Maximum number of doubling attempts before
            giving up with a RuntimeError.
        """
        u = u.clone().to(dtype=dynamics.dtype, device=dynamics.device)
        if u.shape != (dynamics.n_inputs,):
            raise ValueError(
                f"u has bad shape {u.shape}. Expected ({dynamics.n_inputs},)"
            )
        dynamics._validate_u(u)

        if x_init is None:
            if dynamics.x_min is not None and dynamics.x_max is not None:
                x = (dynamics.x_min + dynamics.x_max) / 2.0
            elif dynamics.x_min is not None:
                x = dynamics.x_min + 1.0
            elif dynamics.x_max is not None:
                x = dynamics.x_max - 1.0
            else:
                x = torch.zeros(
                    dynamics.state_size, dtype=dynamics.dtype, device=dynamics.device
                )
        else:
            x = x_init.clone().to(dtype=dynamics.dtype, device=dynamics.device)
            if x.shape != (dynamics.state_size,):
                raise ValueError(
                    f"x has bad shape {x.shape}. Expected ({dynamics.state_size},)"
                )
            dynamics._validate_x(x)

        def u_func(_t: float) -> torch.Tensor:
            return u

        t0_tensor = torch.as_tensor(0.0, dtype=dynamics.dtype, device=dynamics.device)
        duration = initial_duration
        residual_norm = float("inf")
        for _ in range(max_attempts):
            _, x_traj, _ = dynamics.simulate(u_func, duration, initial_state=x)
            x = x_traj[-1]
            with torch.no_grad():
                residual_norm = float(dynamics.f_original(t0_tensor, x, u).norm())
            if residual_norm <= tol:
                y = dynamics._g(x, u)
                dynamics._validate_x(x)
                return OperatingPoint(
                    u, x, y, dtype=dynamics.dtype, device=dynamics.device
                )
            duration *= 2.0

        raise RuntimeError(
            f"Could not find a stable equilibrium for u={u.tolist()} within "
            f"{max_attempts} attempts (final duration={duration / 2:.3e}, "
            f"||f(0, x, u)||={residual_norm:.3e} still exceeds tol={tol:.1e}). "
            "The system may have no stable equilibrium for this input (e.g. a "
            "limit cycle or divergence), or may need a larger "
            "initial_duration, more max_attempts, or a looser tol."
        )


def _nice_format_float(x: float) -> str:
    formated = f"{x:.4g}".rstrip("0").rstrip(".")
    if len(formated) == 0:
        return "0"
    else:
        return formated


def _matrix_latex_str(M: torch.Tensor) -> str:
    M_str = ""
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            M_str += str(_nice_format_float(M[i, j].item()))
            if j != M.shape[1] - 1:
                M_str += " & "
        M_str += r"\\"
    return M_str


class StateSpaceDynamics:
    """Continous-time dynamical system dynamics.

    The system dynamics is described by the following state space equations:
        dx/dt = f(t, x, u)
        y = g(x, u)

    f and g are of type torch.nn.Module.
    x, u, y are assumed tensors of rank 1.

    A StateSpaceDynamics may optionally carry an `OperatingPoint` (u, x, y).
    It is never used to shift what gets passed to f_module/g_module.
    u_func / simulate() / step_response() always pass u straight through to f/g.
    operating_point only supplies defaults (the default `initial_state` for
    simulate(), the default reference point for step_response(), the default
    operating point for regulators).

    u_min/u_max/y_min/y_max are checked against whatever u/y actually flow
    through the f_module/g_module.
    """

    def __init__(
        self,
        f_module: nn.Module,
        g_module: nn.Module,
        n_inputs: int,
        n_outputs: int,
        state_size: int,
        operating_point: Optional[OperatingPoint] = None,
        u_min: Optional[torch.Tensor] = None,
        u_max: Optional[torch.Tensor] = None,
        x_min: Optional[torch.Tensor] = None,
        x_max: Optional[torch.Tensor] = None,
        y_min: Optional[torch.Tensor] = None,
        y_max: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_spectral_radius: Optional[float] = None,
    ):
        """Constructor of the StateSpaceDynamics class.

        @param f_module nn.Module with forward(t, x, u)
        @param g_module nn.Module with forward(x, u)
        @param n_inputs Dimension of input u
        @param n_outputs Dimension of output y
        @param state_size Dimension of the system state x
        @param operating_point Optional default (u, x, y) reference point.
            Can be assigned later via `dynamics.operating_point = op`,
            e.g. after `OperatingPoint.from_input(dynamics, u)`.
        @param u_min Minimal feasible input
        @param u_max Maximal feasible input
        @param x_min Minimal feasible state
        @param x_max maximal feastible state
        @param y_min Minimal feasible output
        @param y_max Maximal feasible output
        @param dtype Data type (defaults to torch.float64)
        @param device Device (defaults to torch.device("cpu"))
        @param max_spectral_radius For LTI systems this is computed exactly and
            this parameter is ignored (see `from_linear`, which overwrites it
            after construction). For NONLINEAR systems, this lets you supply a
            conservative upper bound on the spectral radius of df/dx over the
            system's whole reachable operating envelope (i.e. valid for every
            (t, x, u) with x within [x_min, x_max] and u within [u_min,
            u_max]). It is then used to pick a stable step size and to
            warn when a chosen step exceeds the solver's stability limit.

            This is an ASSERTION you are making about your system, not
            something the package verifies: if the bound you supply is too
            small, the resulting step-size choice/warnings will be
            unreliable.
        """
        # Validate input/output dimensions are positive
        if n_inputs <= 0:
            raise ValueError("n_inputs must be positive")
        if n_outputs <= 0:
            raise ValueError("n_outputs must be positive")
        if state_size <= 0:
            raise ValueError("state_size must be positive")

        # Validate u_min and u_max dimensions
        if u_min is not None and u_min.shape != (n_inputs,):
            raise ValueError(f"u_min has bad shape (should be ({n_inputs},))")
        if u_max is not None and u_max.shape != (n_inputs,):
            raise ValueError(f"u_max has bad shape (should be ({n_inputs},))")
        if u_min is not None and u_max is not None and (u_min > u_max).any():
            raise ValueError("u_min must be <= u_max elementwise")

        # Validate y_min and y_max dimensions
        if y_min is not None and y_min.shape != (n_outputs,):
            raise ValueError(f"y_min has bad shape (should be ({n_outputs},))")
        if y_max is not None and y_max.shape != (n_outputs,):
            raise ValueError(f"y_max has bad shape (should be ({n_outputs},))")
        if y_min is not None and y_max is not None and (y_min > y_max).any():
            raise ValueError("y_min must be <= y_max elementwise")

        # Validate max_spectral_radius: must be a positive plain float if given
        if max_spectral_radius is not None and max_spectral_radius <= 0:
            raise ValueError("max_spectral_radius must be positive")

        # Validate x_min and x_max dimensions
        if x_min is not None and x_min.shape != (state_size,):
            raise ValueError(f"x_min has bad shape (should be ({state_size},)")
        if x_max is not None and x_max.shape != (state_size,):
            raise ValueError(f"x_max has bad shape (should be ({state_size},))")
        if x_min is not None and x_max is not None and (x_min > x_max).any():
            raise ValueError("x_min must be <= x_max elementwise")

        self.dtype = dtype or torch.float64
        self.device = device or torch.device("cpu")

        self.state_size = state_size
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.u_min = (
            u_min.clone().to(dtype=self.dtype, device=self.device)
            if u_min is not None
            else None
        )
        self.u_max = (
            u_max.clone().to(dtype=self.dtype, device=self.device)
            if u_max is not None
            else None
        )
        self.x_min = (
            x_min.clone().to(dtype=self.dtype, device=self.device)
            if x_min is not None
            else None
        )
        self.x_max = (
            x_max.clone().to(dtype=self.dtype, device=self.device)
            if x_max is not None
            else None
        )
        self.y_min = (
            y_min.clone().to(dtype=self.dtype, device=self.device)
            if y_min is not None
            else None
        )
        self.y_max = (
            y_max.clone().to(dtype=self.dtype, device=self.device)
            if y_max is not None
            else None
        )

        # Validate operating_point shape/dtype/device consistency.
        # Does NOT verify the equilibrium condition (this is the
        # job of OperatingPoint's constructor).
        if operating_point is not None:
            if operating_point.u.shape != (n_inputs,):
                raise ValueError(
                    f"operating_point.u has bad shape {operating_point.u.shape}. Expected ({n_inputs},)"
                )
            if operating_point.x.shape != state_size:
                raise ValueError(
                    f"operating_point.x has bad shape {operating_point.x.shape}. Expected ({state_size},)"
                )
            if operating_point.y.shape != (n_outputs,):
                raise ValueError(
                    f"operating_point.y has bad shape {operating_point.y.shape}. Expected ({n_outputs},)"
                )
            operating_point = OperatingPoint(
                operating_point.u,
                operating_point.x,
                operating_point.y,
                dtype=self.dtype,
                device=self.device,
            )
        self.operating_point: Optional[OperatingPoint] = operating_point

        # For LTI systems built via from_linear(), this gets overwritten
        # with the exact value computed from A's eigenvalues.
        self.max_spectral_radius: Optional[float] = max_spectral_radius

        # Is the current system LTI. Gets overwritten by
        # `from_linear` constructor.
        self.is_lti = False

        if max_spectral_radius is not None:
            warnings.warn(
                "max_spectral_radius was supplied for a (potentially) nonlinear "
                "system: this is used as-is to pick a stable step size and to "
                "warn about stability limits, but it is NOT verified by the "
                "package. Make sure it is a genuine upper bound on the "
                "spectral radius of df/dx over the whole operating envelope "
                "you intend to simulate (i.e. valid for every x within "
                "[x_min, x_max] and u within [u_min, u_max]), or step-size "
                "choices and warnings based on it will be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        # Store original f and g modules
        self.f_original = f_module.to(dtype=self.dtype, device=self.device).eval()
        self.g_original = g_module.to(dtype=self.dtype, device=self.device).eval()

        # Validate f and g output shapes
        dummy_t = torch.zeros((), dtype=self.dtype, device=self.device)
        if x_min is not None and x_max is not None:
            dummy_x = (x_min + x_max) / 2.0
        elif x_min is not None:
            dummy_x = x_min + 1.0
        elif x_max is not None:
            dummy_x = x_max - 1.0
        else:
            dummy_x = torch.zeros(self.state_size, dtype=self.dtype, device=self.device)
        if u_min is not None and u_max is not None:
            dummy_u = (u_min + u_max) / 2.0
        elif u_min is not None:
            dummy_u = u_min + 1.0
        elif u_max is not None:
            dummy_u = u_max - 1.0
        else:
            dummy_u = torch.zeros(self.n_inputs, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            dx_example = self.f_original(dummy_t, dummy_x, dummy_u)
            y_example = self.g_original(dummy_x, dummy_u)
        if dx_example.shape != (self.state_size,):
            raise ValueError(
                f"f must return shape {(self.state_size,)}, got {dx_example.shape}"
            )
        if y_example.shape != (self.n_outputs,):
            raise ValueError(
                f"g must return shape {(self.n_outputs,)}, got {y_example.shape}"
            )

    def _check_u(self, u: torch.Tensor) -> bool:
        """Check an input against u_min/u_max."""
        if self.u_min is not None and (u < self.u_min).any():
            return False
        if self.u_max is not None and (u > self.u_max).any():
            return False
        return True

    def _validate_u(self, u: torch.Tensor) -> None:
        """Validate an input against u_min/u_max."""
        if self.u_min is not None and (u < self.u_min).any():
            raise ValueError(
                f"Input {u.tolist()} is below u_min {self.u_min.tolist()}."
            )
        if self.u_max is not None and (u > self.u_max).any():
            raise ValueError(f"Input {u.tolist()} exceeds u_max {self.u_max.tolist()}.")

    def _check_x(self, x: torch.Tensor) -> bool:
        """Check a state (or state trajectory) against x_min/x_max. Works
        for a single state, shape (state_size,), or a trajectory, shape
        (num_points, state_size), via broadcasting.
        """
        if self.x_min is not None and (x < self.x_min).any():
            return False
        if self.x_max is not None and (x > self.x_max).any():
            return False
        return True

    def _validate_x(self, x: torch.Tensor) -> None:
        """Validate a state (or state trajectory) against x_min/x_max. Works
        for a single state, shape (state_size,), or a trajectory, shape
        (num_points, state_size), via broadcasting.
        """
        if self.x_min is not None and (x < self.x_min).any():
            raise ValueError("State (or state trajectory) violates x_min.")
        if self.x_max is not None and (x > self.x_max).any():
            raise ValueError("State (or state trajectory) violates x_max.")

    def _check_y(self, y: torch.Tensor) -> bool:
        """Check an output (or output trajectory) against y_min/y_max,
        via broadcasting, exactly like `_check_x`.
        """
        if self.y_min is not None and (y < self.y_min).any():
            return False
        if self.y_max is not None and (y > self.y_max).any():
            return False
        return True

    def _validate_y(self, y: torch.Tensor) -> None:
        """Validate an output (or output trajectory) against y_min/y_max,
        via broadcasting, exactly like `_validate_x`.
        """
        if self.y_min is not None and (y < self.y_min).any():
            raise ValueError(f"Output violates y_min {self.y_min.tolist()}.")
        if self.y_max is not None and (y > self.y_max).any():
            raise ValueError(f"Output violates y_max {self.y_max.tolist()}.")

    def _g(
        self, x: torch.Tensor, u: torch.Tensor, skip_checks: bool = False
    ) -> torch.Tensor:
        """Evaluate the output equation, validated against y_min/y_max.

        This is the ONLY sanctioned way to evaluate g_original from outside
        this class. The x_min/x_max, u_min/u_max, y_min/y_max checks are
        applied automatically and cannot be skipped by a caller.

        @param x The state x
        @param u The input u
        @param skip_checks Should the bounds checks be skipped

        @return y The output y
        """
        if not skip_checks:
            self._validate_x(x)
            self._validate_u(u)
        y = self.g_original(x, u)
        if not skip_checks:
            self._validate_y(y)
        return y

    def _create_traced_ode_function(self, u: torch.Tensor) -> Callable:
        """Create a traced ODE function with signature (t, x) for mini-ode."""

        def ode_func(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return self.f_original(t, x.reshape([-1]), u)

        example_t = torch.zeros((), dtype=self.dtype, device=self.device)
        if self.x_min is not None and self.x_max is not None:
            example_x = (self.x_min + self.x_max) / 2.0
        elif self.x_min is not None:
            example_x = self.x_min + 1.0
        elif self.x_max is not None:
            example_x = self.x_max - 1.0
        else:
            example_x = torch.zeros(
                self.state_size, dtype=self.dtype, device=self.device
            )
        return torch.jit.trace(ode_func, (example_t, example_x))

    def _integrate_constant_input(
        self,
        x0: torch.Tensor,
        u: torch.Tensor,
        duration: float,
        solver: object,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper for simulating with constant input.

        @param x0 The initial state
        @param u Constant input tensor of shape (n_inputs,)
        @param duration Simulation duration
        @param solver Configured solver

        @return t_points Time points
        @return x_traj State trajectory
        """
        traced_ode_fn = self._create_traced_ode_function(u)
        t_span = (0.0, duration)
        y0 = x0.clone()

        try:
            t_points, x_traj = solver.solve(traced_ode_fn, t_span, y0)
        except Exception as e:
            raise RuntimeError(f"ODE solver failed during integration: {e}")

        return t_points.to(self.dtype), x_traj

    def _compute_outputs(
        self, x_traj: torch.Tensor, u: torch.Tensor, skip_checks: bool = False
    ) -> torch.Tensor:
        """Evaluate outputs along trajectory.

        @param x_traj State trajectory of shape (num_points, state_size)
        @param u Input tensor of shape (n_inputs,)
        @param skip_checks Should the bounds checks be skipped

        @return y_traj Output trajectory of shape (num_points, n_outputs)
        """
        y_traj = torch.empty(
            x_traj.shape[0], self.n_outputs, dtype=self.dtype, device=self.device
        )
        with torch.no_grad():
            for i in range(x_traj.shape[0]):
                x_i = x_traj[i].to(self.dtype)
                y_traj[i] = self._g(x_i, u, skip_checks=skip_checks)
        return y_traj

    def _compute_step_response(
        self,
        duration: float,
        solver: object,
        amp: float,
        operating_point: Optional[OperatingPoint] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute and normalize step response.

        @param duration Simulation duration
        @param solver Pre-configured solver
        @param amp Amplitude of the step input
        @param operating_point The operating point

        @return t_points Equally spaced time points, shape (num_points,)
        @return S Normalized step response matrix, shape (num_points, p, m)
        """
        # Checks that exactly one operating point is provided
        if self.operating_point is None and operating_point is None:
            raise RuntimeError(
                "Step response calculation requires a known operating point"
            )
        if self.operating_point is not None and operating_point is not None:
            raise RuntimeError(
                "Two operating points were provided. Step response calculation only works with one operating point."
            )

        # Determine and validate the operating point
        if operating_point is not None:
            # Basic feasibility checks, stability of the operating point is not re-validated.
            self._validate_u(operating_point.u)
            self._validate_x(operating_point.x)
            self._validate_y(operating_point.y)
        else:
            operating_point = self.operating_point

        u0 = operating_point.u
        x0 = operating_point.x

        # Check that the step fits in feasibility region
        # Note: u0+amp is never actually used as input, but this works
        #       instead of validating each input index step because bounds
        #       are intervals
        # Note: For LTI systems we can skip input validation
        #       for computing step response as it only
        #       matters for enforcing model behaviour in
        #       simulation
        if not self.is_lti and not self._check_u(u0 + amp):
            raise ValueError("Amplitude does not fit in input limits of the system")

        # Get initial output
        original_y = self._g(x0, u0)

        S_list: List[torch.Tensor] = []
        t_points: Optional[torch.Tensor] = None
        for j in range(self.n_inputs):
            u_step = u0.clone()
            u_step[j] += amp
            t_points_j, x_traj = self._integrate_constant_input(
                x0, u_step, duration, solver
            )

            # We perform x bounds check only for systems that are not LTI - this is not simulation
            if not self.is_lti and not self._check_x(x_traj):
                raise ValueError("State exceeded limits of the system")

            if t_points is None:
                t_points = t_points_j
            elif not torch.allclose(t_points, t_points_j):
                raise RuntimeError("Time points mismatch across input channels")

            y_traj = self._compute_outputs(x_traj, u_step, skip_checks=self.is_lti)
            S_list.append(y_traj)

        S = torch.stack(S_list, dim=-1)  # (num_points, p, m)
        assert S.ndim == 3
        assert original_y.ndim == 1
        S = S - original_y.unsqueeze(dim=1).repeat(1, S.shape[2])

        return t_points, S / amp

    def step_response(
        self,
        duration: float,
        step_size: Optional[float] = None,
        solver: Optional[object] = None,
        amp: float = 1.0,
        operating_point: Optional[OperatingPoint] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute MIMO step response matrix using mini-ode.

        @param duration Simulation duration
        @param step_size Fixed integration step (default: computed from
                         spectral radius and solver stability radius)
        @param solver Fixed-step mini-ode solver (default: RK4MethodSolver)
        @param amp Amplitude of the step input
        @param operating_point Operating point

        @return t_points Equally spaced time points, shape (num_points,)
        @return S Step response matrix, shape (num_points, p, m)
        """
        # Validate amp to prevent division by zero
        if amp == 0:
            raise ValueError("amp must not be zero (would cause division by zero)")

        step_size, solver = resolve_default_solver(
            duration,
            step_size,
            solver,
            max_spectral_radius=self.max_spectral_radius,
            require_fixed_step=True,
            discrete=False,
        )

        return self._compute_step_response(duration, solver, amp, operating_point)

    def discrete_step_response(
        self,
        duration: float,
        dt: float,
        solver: Optional[object] = None,
        amp: float = 1.0,
        operating_point: Optional[OperatingPoint] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute MIMO discrete step response matrix using mini-ode.

        @param duration Simulation duration
        @param dt Fixed time step for equal spacing
        @param solver Fixed-step mini-ode solver (default: RK4MethodSolver)
        @param amp Amplitude of the step input
        @param operating_point Operating point

        @return t_points Equally spaced time points, shape (num_points,)
        @return S Step response matrix, shape (num_points, p, m)
        """
        if duration <= 0 or dt <= 0:
            raise ValueError("Duration and dt must be positive")
        if amp == 0:
            raise ValueError("amp must not be zero (would cause division by zero)")

        _, solver = resolve_default_solver(
            duration,
            dt,
            solver,
            max_spectral_radius=self.max_spectral_radius,
            require_fixed_step=True,
            discrete=True,
        )

        if not math.isclose(solver.step, dt, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(
                f"The solver's step size {solver.step} "
                f"does not match the provided dt {dt}."
            )

        return self._compute_step_response(duration, solver, amp, operating_point)

    def simulate(
        self,
        u_func: Callable[[float], torch.Tensor],
        duration: float,
        initial_state: Optional[torch.Tensor] = None,
        step_size: Optional[float] = None,
        solver: Optional[object] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate system with time-varying input u(t).

        @param u_func Function u(t) that returns input at time t, shape (m,)
        @param duration Simulation duration
        @param initial_state Starting state (defaults to operating point's initial state x)
        @param step_size Time step for updating the input (default: computed
                         from spectral radius and solver stability radius)
        @param solver mini-ode solver object (default: RKF45MethodSolver)

        @return t_points Time points (absolute timestamps after shifting by StateSpaceSystem.simulation_time)
        @return x_traj State trajectory, shape (length, state_size)
        @return y_traj Output trajectory, shape (length, n_outputs)

        Note: x_traj[i] corresponds to t_points[i]. In StateSpaceSystem.simulate(),
              t_points will be shifted to reflect accumulated simulation time.
        """
        step_size, solver = resolve_default_solver(
            duration,
            step_size,
            solver,
            max_spectral_radius=self.max_spectral_radius,
            require_fixed_step=False,
            discrete=False,
        )

        # Create timepoints
        times = torch.arange(
            0, duration, step_size, dtype=self.dtype, device=self.device
        )

        # Append duration if not within tolerance
        if not torch.isclose(
            times[-1], times.new_tensor([duration]), rtol=1e-12, atol=1e-15
        ):
            times = torch.cat([times, times.new_tensor([duration])])

        t_points = times

        num_steps = t_points.shape[0]
        x_traj = torch.empty(
            num_steps, self.state_size, dtype=self.dtype, device=self.device
        )
        y_traj = torch.empty(
            num_steps, self.n_outputs, dtype=self.dtype, device=self.device
        )

        # Validate and properly handle initial_state
        if initial_state is not None:
            if initial_state.shape != (self.state_size,):
                raise ValueError(
                    f"initial_state must have shape {(self.state_size,)}. "
                    f"Got {initial_state.shape}"
                )
            current_x = initial_state.to(dtype=self.dtype, device=self.device).clone()
        elif self.operating_point is not None:
            current_x = self.operating_point.x.clone()
        else:
            raise RuntimeError("Initial system state to begin simulation not provided.")

        x_traj[0] = current_x

        u_val = u_func(t_points[0].item())
        if u_val.shape != (self.n_inputs,):
            raise ValueError(
                f"u_func must return shape {(self.n_inputs,)}, got {u_val.shape}"
            )
        u_tensor = u_val.to(dtype=self.dtype, device=self.device)
        y_traj[0] = self._g(current_x, u_tensor)

        # Retracing is necessary when u changes between steps
        with torch.no_grad():
            for i in range(num_steps - 1):
                if not self._check_u(u_tensor):
                    raise ValueError(
                        f"Current input {u_tensor.tolist()} does not fit in input limits of the system (min: {self.u_min.tolist()}, max: {self.u_max.tolist()})"
                    )

                # Create traced ODE with current u
                traced_ode_fn = self._create_traced_ode_function(u_tensor)

                # Integrate over [t_i, t_{i+1}]
                t_span = (t_points[i].item(), t_points[i + 1].item())
                y0 = current_x
                try:
                    t_sub, x_sub = solver.solve(traced_ode_fn, t_span, y0)
                except Exception as e:
                    raise RuntimeError(f"ODE solver failed at step {i}: {e}")

                # Update current_x to the value at t_{i+1}
                current_x = x_sub[-1].to(self.dtype)
                x_traj[i + 1] = current_x

                if not self._check_x(current_x):
                    raise ValueError("State exceeded limits of the system")

                # Compute next u and y
                u_val = u_func(t_points[i + 1].item())
                if u_val.shape != (self.n_inputs,):
                    raise ValueError(
                        f"u_func must return shape {(self.n_inputs,)}. "
                        f"Got {u_val.shape}"
                    )
                u_tensor = u_val.to(dtype=self.dtype, device=self.device)
                y_traj[i + 1] = self._g(current_x, u_tensor)

        return t_points, x_traj, y_traj

    @staticmethod
    def from_linear(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        operating_point_u: Optional[torch.Tensor] = None,
        operating_point_y: Optional[torch.Tensor] = None,
        u_min: Optional[torch.Tensor] = None,
        u_max: Optional[torch.Tensor] = None,
        x_min: Optional[torch.Tensor] = None,
        x_max: Optional[torch.Tensor] = None,
        y_min: Optional[torch.Tensor] = None,
        y_max: Optional[torch.Tensor] = None,
        skip_controllability_check: bool = False,
        skip_observability_check: bool = False,
    ) -> "StateSpaceDynamics":
        """Convenience constructor for linear systems.

        A linear system is defined by linear state space equations:
            dx/dt = A x + B u
            y = C x + D u

        @param A State matrix
            Shape: (state_size, state_size)
        @param B Input matrix
            Shape: (state_size, n_inputs)
        @param C Output matrix
            Shape: (n_outputs, state_size)
        @param D Feedthrough matrix
            Shape: (n_outputs, n_inputs)
        @param operating_point_u Operating point input u.
        @param operating_point_y Operating point output y.
        @param u_min Minimal feasible input
        @param u_max Maximal feasible input
        @param x_min Minimal feasible state
        @param x_max maximal feastible state
        @param y_min Minimal feasible output
        @param y_max Maximal feasible output
        @param skip_controllability_check Skip controllability check
        @param skip_observability_check Skip observability check
        """
        # Validation: Reject complex matrices
        if torch.is_complex(A):
            raise ValueError(
                "Complex state-space matrices are not supported. Use real-valued matrices only."
            )
        if torch.is_complex(B):
            raise ValueError(
                "Complex state-space matrices are not supported. Use real-valued matrices only."
            )
        if torch.is_complex(C):
            raise ValueError(
                "Complex state-space matrices are not supported. Use real-valued matrices only."
            )
        if torch.is_complex(D):
            raise ValueError(
                "Complex state-space matrices are not supported. Use real-valued matrices only."
            )

        # Validation of A, B, C, D shapes
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square matrix")
        n = A.shape[0]
        if B.ndim != 2 or B.shape[0] != n:
            raise ValueError(f"B must be ({n}, n_inputs). Got {B.shape}")
        if C.ndim != 2 or C.shape[1] != n:
            raise ValueError(f"C must be (n_outputs, {n}). Got {C.shape}")
        n_inputs = B.shape[1]
        n_outputs = C.shape[0]
        if D.ndim != 2 or D.shape != (n_outputs, n_inputs):
            raise ValueError(f"D must be {(n_outputs, n_inputs)}. Got {D.shape}")

        system_dtype = A.dtype
        system_device = A.device

        # Consolidate dtype/device checks
        for M, name in [(B, "B"), (C, "C"), (D, "D")]:
            if M.dtype != system_dtype or M.device != system_device:
                raise ValueError(
                    f"All matrices must have the same datatype and device. '{name}' differs from 'A'."
                )

        # Controllability check
        if not skip_controllability_check and (
            torch.linalg.matrix_rank(
                torch.cat(
                    [torch.linalg.matrix_power(A, i) @ B for i in range(n)], dim=1
                )
            )
            < n
        ):
            raise ValueError(f"System is not controllable.")

        # Observability check
        if not skip_observability_check and (
            torch.linalg.matrix_rank(
                torch.cat(
                    [C @ torch.linalg.matrix_power(A, i) for i in range(n)], dim=0
                )
            )
            < n
        ):
            raise ValueError(f"System is not observable.")

        if operating_point_u is None:
            operating_point_u = torch.zeros(
                (n_inputs,), dtype=system_dtype, device=system_device
            )
        if operating_point_y is None:
            operating_point_y = torch.zeros(
                (n_outputs,), dtype=system_dtype, device=system_device
            )
        operating_point_x = torch.zeros(n, dtype=system_dtype, device=system_device)
        operating_point = OperatingPoint(
            operating_point_u, operating_point_x, operating_point_y
        )

        # Use class-level LinearF/LinearG instead of nested classes
        f_mod = LinearF(A, B, u0=operating_point_u)
        g_mod = LinearG(C, D, u0=operating_point_u, y0=operating_point_y)

        system_dynamics = StateSpaceDynamics(
            f_mod,
            g_mod,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            state_size=n,
            u_min=u_min,
            u_max=u_max,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            dtype=system_dtype,
            device=system_device,
        )

        # Proper spectral radius computation as float with eigvals
        eigvals = torch.linalg.eigvals(A)
        spectral_radius = float(eigvals.abs().max())

        system_dynamics.max_spectral_radius = spectral_radius
        system_dynamics.is_lti = True
        system_dynamics.operating_point = operating_point

        return system_dynamics

    @staticmethod
    def from_tf(
        H: sp.Matrix,
        s: sp.var,
        operating_point_u: Optional[torch.Tensor] = None,
        operating_point_y: Optional[torch.Tensor] = None,
        u_min: Optional[torch.Tensor] = None,
        u_max: Optional[torch.Tensor] = None,
        x_min: Optional[torch.Tensor] = None,
        x_max: Optional[torch.Tensor] = None,
        y_min: Optional[torch.Tensor] = None,
        y_max: Optional[torch.Tensor] = None,
    ) -> "StateSpaceDynamics":
        """Convert transfer function matrix to state space form.

        @param H Transfer function matrix (Sympy Matrix)
        @param s Laplace variable (Sympy symbol)
        @param operating_point_u Operating point input u.
        @param operating_point_y Operating point output y.
        @param u_min Minimal feasible input
        @param u_max Maximal feasible input
        @param x_min Minimal feasible state
        @param x_max maximal feastible state
        @param y_min Minimal feasible output
        @param y_max Maximal feasible output
        """
        # Sympify H
        H_sp = H.applyfunc(lambda e: sp.sympify(e))

        # Add validation checks
        if H_sp.rows == 0 or H_sp.cols == 0:
            raise ValueError("H must be a non-empty matrix.")

        for i in range(H_sp.rows):
            for j in range(H_sp.cols):
                entry = H_sp[i, j]
                # Use proper warning category
                if entry.has(sp.Float):
                    warnings.warn(
                        "Transfer function matrix contains Float entries "
                        "that may compromise numerical quality.",
                        RuntimeWarning,
                    )

                # Check for invalid denominators
                if entry.is_rational_function(s):
                    denom = sp.denom(entry)
                    if denom == 0:
                        raise ValueError(f"H[{i},{j}] has zero denominator")

        # Handle tf2ss return values appropriately
        tf_result = tf2ss(H_sp, s)
        if len(tf_result) == 4:
            A_sp, B_sp, C_sp, D_sp = tf_result
            D = torch.tensor(np.array(D_sp, dtype=np.float64))
        else:
            A_sp, B_sp, C_sp = tf_result
            D_shape = (C_sp.shape[0], B_sp.shape[1])
            D = torch.zeros(D_shape, dtype=torch.float64)

        # Safer large fraction detection with is_number check
        for mat, name in [(A_sp, "A"), (B_sp, "B"), (C_sp, "C")]:
            if mat is not None:
                for idx in np.ndindex(mat.shape):
                    val = mat[idx]
                    if hasattr(val, "as_numer_denom"):
                        num, den = val.as_numer_denom()
                        if num.is_number and den.is_number:
                            num_val = abs(float(num))
                            den_val = abs(float(den))
                            if num_val > 1e6 or den_val > 1e6:
                                warnings.warn(
                                    f"Large coefficients detected in {name}[{idx}], "
                                    "consider normalizing the transfer function.",
                                    RuntimeWarning,
                                )
                                break

        A = torch.tensor(np.array(A_sp, dtype=np.float64))
        B = torch.tensor(np.array(B_sp, dtype=np.float64))
        C = torch.tensor(np.array(C_sp, dtype=np.float64))

        return StateSpaceDynamics.from_linear(
            A,
            B,
            C,
            D,
            operating_point_u=operating_point_u,
            operating_point_y=operating_point_y,
            u_min=u_min,
            u_max=u_max,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

    def _repr_latex_(self) -> str:
        if self.is_lti:
            A_str = _matrix_latex_str(self.f_original.A)
            B_str = _matrix_latex_str(self.f_original.B)
            C_str = _matrix_latex_str(self.g_original.C)
            D_str = _matrix_latex_str(self.g_original.D)
            return (
                r"$\begin{gather}\dot{x} = \begin{bmatrix}"
                + A_str
                + r"\end{bmatrix} x + \begin{bmatrix}"
                + B_str
                + r"\end{bmatrix} u \\ \\ y = \begin{bmatrix}"
                + C_str
                + r"\end{bmatrix} x + \begin{bmatrix}"
                + D_str
                + r"\end{bmatrix} u\end{gather}$"
            )
        else:
            return r"$\begin{gather}\dot{x} = F(t, x, u) \\ y = G(t, x, u)\end{gather}$"


class StateSpaceSystem:
    """Wrapper for StateSpaceDynamics with mutable state and time tracking.

    Note on timing: The StateSpaceSystem maintains cumulative simulation time.
    When simulate() is called:
    - t_points are shifted by simulation_time to give absolute timestamps
    - x_traj and y_traj remain relative to the current simulation state
    - After simulate(), x stores the final state for continuation
    - reset() clears both state AND simulation_time to 0
    """

    def __init__(
        self,
        dynamics: StateSpaceDynamics,
        x: Optional[torch.Tensor] = None,
    ):
        self.dynamics = dynamics

        self.dtype = self.dynamics.dtype
        self.device = self.dynamics.device
        self.n_inputs = self.dynamics.n_inputs
        self.n_outputs = self.dynamics.n_outputs
        self.state_size = self.dynamics.state_size

        if x is not None:
            if x.shape != (self.state_size,):
                raise ValueError(
                    f"`x` must have shape {(self.state_size,)}. " f"Got {x.shape}"
                )
            self.x = x.to(dtype=self.dtype, device=self.device)
        elif dynamics.operating_point is not None:
            self.x = dynamics.operating_point.x.clone().to(
                dtype=self.dtype, device=self.device
            )
        else:
            raise RuntimeError("System state x not provided.")

        self.simulation_time: float = 0.0

    def reset(
        self,
        x: Optional[torch.Tensor] = None,
    ) -> None:
        """Reset internal state and simulation time.

        @param x Optional new initial state (defaults to dynamics.x0)
        """
        if x is not None:
            if x.shape != (self.state_size,):
                raise ValueError(
                    f"`x` must have shape {(self.state_size,)}. " f"Got {x.shape}"
                )
            self.x = x.to(dtype=self.dtype, device=self.device)
        elif self.dynamics.operating_point is not None:
            self.x = self.dynamics.operating_point.x.clone().to(
                dtype=self.dtype, device=self.device
            )
        else:
            raise RuntimeError("System state x not provided.")

        # Reset simulation time to 0
        self.simulation_time = 0.0

    def simulate(
        self,
        u_func: Callable[[float], torch.Tensor],
        duration: float,
        step_size: Optional[float] = None,
        solver: Optional[object] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simulate the system with time-varying input u(t).

        IMPORTANT: Returned t_points are ABSOLUTE timestamps (shifted by
        current simulation_time). The trajectories x_traj and y_traj correspond
        to these timestamps, with x_traj[0] matching the current self.x state.

        @param u_func Function u(t) that returns input at time t, shape (m,)
        @param duration Simulation duration
        @param step_size Time step for updating the input
        @param solver mini-ode solver object

        @return t_points Absolute time points (shifted by simulation_time)
        @return x_traj State trajectory, shape (length, state_size)
        @return y_traj Output trajectory, shape (length, n_outputs)

        After simulation:
        - self.x contains the final state (at t_points[-1])
        - self.simulation_time equals t_points[-1]

        To continue from current state, call simulate() again with a new u_func.
        To restart from initial conditions, call reset() first.
        """

        start_time = self.simulation_time

        def absolute_u_func(t: float) -> torch.Tensor:
            return u_func(t + start_time)

        t_points_rel, x_traj, y_traj = self.dynamics.simulate(
            absolute_u_func,
            duration,
            initial_state=self.x,
            step_size=step_size,
            solver=solver,
        )

        self.x = x_traj[-1].clone()

        t_points_abs = t_points_rel + start_time

        self.simulation_time = float(t_points_abs[-1].item())

        return t_points_abs, x_traj, y_traj

    def __repr__(self) -> str:
        return f"StateSpaceSystem(dynamics=..., x={self.x})"
