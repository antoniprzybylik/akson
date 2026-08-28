import torch
import warnings
from .state_space_system import StateSpaceSystem
from ._utils import resolve_default_solver
from typing import Optional, Tuple
from abc import ABC, abstractmethod
from ._validation import validate_tensor_shape_with_names, validate_tensor


def build_dynamic_matrices(
    step_response: torch.Tensor,
    N: int,
    Nu: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """! Builds dynamic matrix M and matrix of past dynamics Mp for DMC and QDMC controllers.

    @param step_response Discrete step response
        Shape: (D, n_outputs, n_inputs)
    @param N Prediction horizon
    @param Nu Control horizon

    @return M Dynamic matrix
        Shape: (N*n_outputs, Nu*n_inputs)
    @return Mp Matrix of past dynamics
        Shape: (N*n_outputs, (D-1)*n_inputs)
    """
    S = step_response
    D = S.shape[0]
    n_outputs = S.shape[1]
    n_inputs = S.shape[2]
    dtype = S.dtype
    device = S.device

    M = torch.zeros(N * n_outputs, Nu * n_inputs, dtype=dtype, device=device)
    for i in range(1, N + 1):
        for j in range(1, Nu + 1):
            k = i - j + 1
            if k >= 0:
                rows_slice = slice((i - 1) * n_outputs, i * n_outputs)
                columns_slice = slice((j - 1) * n_inputs, j * n_inputs)
                M[rows_slice, columns_slice] = S[min(k, D - 1)]

    Mp = torch.zeros(N * n_outputs, (D - 1) * n_inputs, dtype=dtype, device=device)
    for i in range(1, D):
        for j in range(1, N + 1):
            rows_slice = slice((j - 1) * n_outputs, j * n_outputs)
            columns_slice = slice((i - 1) * n_inputs, i * n_inputs)
            Mp[rows_slice, columns_slice] = S[min(i + j, D - 1)] - S[i]

    return M, Mp


def build_sum_input_deltas_array(
    N: int,
    Nu: int,
    n_inputs: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """! Constructs matrix that sums input signal deltas
    The sum_input_deltas_array is a matrix that can be used to create
    the prefix sums of control increments (du).

    @return sum_input_deltas_array
        Shape: (N*n_inputs, Nu*n_inputs)
    """
    I = torch.eye(n_inputs, dtype=dtype, device=device)
    Z = torch.zeros((n_inputs, n_inputs), dtype=dtype, device=device)
    rows = []
    for i in range(N):
        row = torch.cat([I if i >= j else Z for j in range(Nu)], dim=1)
        rows.append(row)
    return torch.cat(rows, dim=0)


def zero_past_du_and_current_u(
    dynamics_horizon: int,
    n_inputs: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    past_du = torch.zeros(dynamics_horizon - 1, n_inputs, dtype=dtype, device=device)
    current_u = torch.zeros(n_inputs, dtype=dtype, device=device)
    return past_du, current_u


class BaseDMCState(ABC):
    """! Base class for DMC family controller states."""

    def __init__(
        self,
        past_du: torch.Tensor,
        current_u: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        if past_du.ndim != 2:
            raise ValueError(
                f"past_du has bad shape {past_du.shape}. Expected two dimensions."
            )
        if current_u.ndim != 1:
            raise ValueError(
                f"current_u has bad shape {current_u.shape}. Expected one dimension."
            )
        if past_du.shape[1] != current_u.shape[0]:
            raise ValueError("current_u and past_du dimensions do not conform.")

        if dtype is not None:
            past_du = past_du.to(dtype)
            current_u = current_u.to(dtype)
        if device is not None:
            past_du = past_du.to(device)
            current_u = current_u.to(device)

        self.past_du = past_du
        self.current_u = current_u
        self.dtype = past_du.dtype
        self.device = past_du.device


class BaseDMCClosedSystem(ABC):
    """! Base class for DMC family controller closed systems."""

    def __init__(
        self,
        plant_dynamics: "StateSpaceDynamics",
        controller_configuration: (
            "DMCControllerConfiguration" | "QDMCControllerConfiguration"
        ),
        controller_state: "DMCControllerState" | "QDMCControllerState",
    ):
        """! Constructor.

        @param plant_dynamics Plant dynamics
        @param Controller_configuration Controller configuration
        @param controller_state Controller state
        """
        if controller_state.past_du.shape[0] != controller_configuration.D - 1:
            raise ValueError(
                "Controller state and controller configuration do not conform. Different assumed dynamics horizon."
            )
        if controller_state.past_du.shape[1] != controller_configuration.n_inputs:
            raise ValueError(
                "Controller state and controller configuration do not conform. Different assumed number of system inputs."
            )
        if controller_state.current_u.shape[0] != controller_configuration.n_inputs:
            raise ValueError(
                "Controller state and controller configuration do not conform. Different assumed number of system inputs."
            )

        if plant_dynamics.n_inputs != controller_configuration.n_inputs:
            raise ValueError(
                "Plant dynamics and controller configuration do not conform. Different assumed number of system inputs."
            )
        if plant_dynamics.n_outputs != controller_configuration.n_outputs:
            raise ValueError(
                "Plant dynamics and controller configuration do not conform. Different assumed number of system outputs."
            )
        validate_tensor(
            controller_configuration.operating_point.x,
            "controller_configuration.operating_point.x",
            (plant_dynamics.state_size,),
        )

        if (
            plant_dynamics.device != controller_configuration.device
            or controller_configuration.device != controller_state.device
        ):
            raise ValueError("Devices do not match.")
        self.device = plant_dynamics.device
        if (
            plant_dynamics.dtype != controller_configuration.dtype
            or controller_configuration.dtype != controller_state.dtype
        ):
            raise ValueError("Datatypes do not match.")
        self.dtype = plant_dynamics.dtype

        if (
            plant_dynamics.u_min is not None
            and controller_configuration.u_min is not None
            and (plant_dynamics.u_min > controller_configuration.u_max).any()
        ) or (
            plant_dynamics.u_min is not None and controller_configuration.u_min is None
        ):
            warnings.warn(
                "Constraints on u_min are looser in the controller than in the plant. This may lead to controller feeding infeasible input into the plant",
                RuntimeWarning,
            )

        if (
            plant_dynamics.u_max is not None
            and controller_configuration.u_max is not None
            and (plant_dynamics.u_max > controller_configuration.u_max).any()
        ) or (
            plant_dynamics.u_max is not None and controller_configuration.u_max is None
        ):
            warnings.warn(
                "Constraints on u_max are looser in the controller than in the plant. This may lead to controller feeding infeasible input into the plant",
                RuntimeWarning,
            )

        self.plant = StateSpaceSystem(
            plant_dynamics, x=controller_configuration.operating_point.x
        )
        self.config = controller_configuration
        self.state = controller_state

    @abstractmethod
    def step(self, y: torch.Tensor, r_traj: torch.Tensor) -> torch.Tensor:
        """Compute next control input u given current measurement y and reference trajectory.

        @param y Current measured output
            Shape: (n_outputs,)
        @param r_traj Desired trajectory over prediction horizon
            Shape: (N, n_outputs)

        @return u Next control input
            Shape: (n_inputs,)
        """
        pass

    def simulate(
        self,
        r_traj: torch.Tensor,
        duration: float,
        dt: float,
        num_substeps: Optional[int] = None,
        solver: Optional[object] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """! Simulate a closed-loop system.

        @param r_traj Reference trajectory
            Shape: (length, n_outputs). If shorter than needed, extended with last element
        @param duration Total simulation time
        @param dt Time step for controller updates
        @param num_substeps Number of substeps per dt (increase to improve trajectory resolution)
        @param solver mini-ode solver object

        @return t_all Tensor of all time points
            Shape: (num_points,)
        @return y_all Tensor of output values at each time point
            Shape: (num_points, n_outputs)
        @return u_all Tensor of control inputs
            Shape: (num_points, n_inputs)
        """
        # Validation
        if num_substeps < 1:
            raise ValueError("num_substeps must be at least 1")
        validate_tensor_shape_with_names(
            r_traj,
            "r_traj",
            (None, self.plant.n_outputs),
            ("length", None),
            ("length of the reference trajectory", None),
        )

        # Solver step size
        if num_substeps is not None:
            step_size = dt / num_substeps
        else:
            step_size = None

        # Resolve the ODE solver
        step_size, solver = resolve_default_solver(
            duration,
            step_size,
            solver,
            max_spectral_radius=self.plant.dynamics.max_spectral_radius,
            require_fixed_step=False,
            discrete=False,
        )

        # Number of controller control computation steps
        num_steps = int(duration / dt)

        # Extend / cut the reference trajectory
        ref_trajectory_points_needed = num_steps + self.config.N
        r_full = r_traj.to(dtype=self.dtype, device=self.device)
        if r_full.shape[0] < ref_trajectory_points_needed:
            pad_len = ref_trajectory_points_needed - r_full.shape[0]
            last = r_full[-1:].repeat(pad_len, 1)
            r_full = torch.cat([r_full, last], dim=0)
        else:
            r_full = r_full[:ref_trajectory_points_needed]

        t_all = []
        y_all = []
        u_all = []
        current_x = self.plant.x.clone()
        current_t = self.plant.simulation_time
        u = self.config.operating_point.u.clone()
        self.plant.dynamics._validate_u(u)
        y = self.plant.dynamics._g(current_x, u)
        t_all.append(current_t)
        y_all.append(y)
        u_all.append(u)
        for i in range(num_steps):
            # Get r_traj for this step
            r_traj_step = r_full[i + 1 : i + 1 + self.config.N]

            # Calculate control input from the controller
            u_new = self.step(y, r_traj_step)
            self.plant.dynamics._validate_u(u_new)

            # Simulate the system on current timespan
            traced_ode_fn = self.plant.dynamics._create_traced_ode_function(u_new)
            t_span = (current_t, current_t + dt)
            y0 = current_x.to(torch.float64)
            t_sub, x_sub = solver.solve(traced_ode_fn, t_span, y0)
            self.plant.dynamics._validate_x(x_sub)

            # Compute y at each sub-point (validated against y_min/y_max in _g)
            y_sub = [
                self.plant.dynamics._g(x_sub[j].to(self.dtype), u_new)
                for j in range(len(t_sub))
            ]

            # Append new sub-points. Exclude the first, which is
            # the same as previous one
            t_all.extend(t_sub[1:].tolist())
            y_all.extend(y_sub[1:])
            u_all.append(u_new)

            # Update for next step
            current_x = x_sub[-1].to(self.dtype)
            current_t = t_sub[-1].item()
            y = y_sub[-1]

        self.plant.x = current_x
        self.plant.simulation_time = current_t

        return (
            torch.tensor(t_all, dtype=self.dtype, device=self.device),
            torch.stack(y_all),
            torch.stack(u_all),
        )

    @abstractmethod
    def reset(self) -> None:
        """Reset the plant state and the controller state."""
        pass
