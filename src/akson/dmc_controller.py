import torch
import mini_ode
from typing import Optional, Tuple
from .state_space_system import StateSpaceSystem, StateSpaceDynamics, OperatingPoint
from ._utils import resolve_default_solver
from ._dmc_common import (
    build_dynamic_matrices,
    build_sum_input_deltas_array,
    zero_past_du_and_current_u,
    BaseDMCState,
    BaseDMCClosedSystem,
)
from ._validation import (
    validate_tensor_shape_with_names,
    validate_tensor,
    validate_and_move_optional_tensor,
    validate_optional_tensors_le,
)


class DMCControllerConfiguration:
    """! Dynamic Matrix Control (DMC) controller configuration."""

    def __init__(
        self,
        step_response: torch.Tensor,
        N: int,
        Nu: int,
        operating_point: OperatingPoint,
        regularisation: float = 0.0,
        du_max: Optional[float] = None,
        du_min: Optional[float] = None,
        u_max: Optional[torch.Tensor] = None,
        u_min: Optional[torch.Tensor] = None,
        use_polishing: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """! Constructor of the DMCControllerConfiguration class

        @param step_response Discrete step response tensor
            Shape: (length, n_outputs, n_inputs)
        @param N Prediction horizon
        @param Nu Control horizon
        @param operating_point Stable operating point (u0, x0, y0)
        @param regularisation Regularisation factor
        @param du_min Minimum allowed control increment
            Shape: (n_inputs,)
        @param du_max Maximum allowed control increment
            Shape: (n_inputs,)
        @param u_min Minimum allowed control signal
            Shape: (n_inputs,)
        @param u_max Maximum allowed control signal
            Shape: (n_inputs,)
        @param use_polishing Enables 'Active-Set' solution polishing
        @param dtype Torch dtype
        @param device Torch device
        """
        # Validation
        validate_tensor_shape_with_names(
            step_response,
            "step_response",
            (None, None, None),
            ("length", "n_outputs", "n_inputs"),
            (
                "length of step response",
                "number of system outputs",
                "number of system inputs",
            ),
        )
        if step_response.shape[0] < 1:
            raise ValueError("step_response must not contain less than one sample")
        if step_response.shape[1] < 1:
            raise ValueError(
                f"System output size must not be smaller than one. Got shape {step_response.shape}"
            )
        if step_response.shape[2] < 1:
            raise ValueError(
                f"System input size must not be smaller than one. Got shape {step_response.shape}"
            )
        if N < 1:
            raise ValueError("Prediction horizon N must be a positive integer")
        if Nu < 1:
            raise ValueError("Control horizon Nu must be a positive integer")
        if regularisation < 0:
            raise ValueError("Regularisation factor must be nonnegative")

        self.S = step_response.clone()
        if dtype is not None:
            self.S = self.S.to(dtype)
        if device is not None:
            self.S = self.S.to(device)

        self.D = self.S.shape[0]  # Dynamics horizon
        self.n_inputs = self.S.shape[2]  # Number of system inputs
        self.n_outputs = self.S.shape[1]  # Number of system outputs
        self.N = N  # Prediction horizon
        self.Nu = Nu  # Control horizon
        self.regularisation = regularisation  # Regularisation factor

        self.use_polishing = use_polishing  # Enables 'Active-Set' solution polishing

        self.dtype = self.S.dtype
        self.device = self.S.device

        # Validate operating point tensor shapes
        validate_tensor(operating_point.u, "operating point input", (self.n_inputs,))
        validate_tensor(operating_point.y, "operating point output", (self.n_outputs,))
        self.operating_point = OperatingPoint(
            u=operating_point.u,
            x=operating_point.x,
            y=operating_point.y,
            device=self.device,
            dtype=self.dtype,
        )

        # Validate and store bounds
        self.du_min = validate_and_move_optional_tensor(
            du_min,
            "du_min",
            (self.n_inputs,),
            desired_dtype=self.dtype,
            desired_device=self.device,
        )
        self.du_max = validate_and_move_optional_tensor(
            du_max,
            "du_max",
            (self.n_inputs,),
            desired_dtype=self.dtype,
            desired_device=self.device,
        )
        validate_optional_tensors_le(self.du_min, "du_min", self.du_max, "du_max")
        self.u_min = validate_and_move_optional_tensor(
            u_min,
            "u_min",
            (self.n_inputs,),
            desired_dtype=self.dtype,
            desired_device=self.device,
        )
        self.u_max = validate_and_move_optional_tensor(
            u_max,
            "u_max",
            (self.n_inputs,),
            desired_dtype=self.dtype,
            desired_device=self.device,
        )
        validate_optional_tensors_le(self.u_min, "u_min", self.u_max, "u_max")

        # Build and store dynamic matrices
        self.M, self.Mp = build_dynamic_matrices(self.S, self.N, self.Nu)

        # Compute K = (M^T*M + lambda*I)^-1 M^T
        MTM = self.M.T @ self.M
        Lambda = self.regularisation * torch.eye(
            self.Nu * self.n_inputs, dtype=self.dtype, device=self.device
        )
        self.K = torch.linalg.solve(MTM + Lambda, self.M.T)

        # Build and store "sum input deltas" array
        self.sum_input_deltas_array = build_sum_input_deltas_array(
            self.N, self.Nu, self.n_inputs, self.dtype, self.device
        )


class DMCControllerState(BaseDMCState):
    @staticmethod
    def zero_state(
        dynamics_horizon: int,
        n_inputs: int,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ) -> DMCControllerState:
        """! Constructs zero state for the DMC controller.

        @param dynamics_horizon Dynamics horizon length
        @param n_inputs Number of inputs to the system
        @return controller_state DMC controller state
        """
        past_du, current_u = zero_past_du_and_current_u(
            dynamics_horizon, n_inputs, dtype, device
        )
        return DMCControllerState(past_du, current_u, dtype=dtype, device=device)

    @staticmethod
    def initial_state_for(
        controller_configuration: DMCControllerConfiguration,
    ) -> DMCControllerState:
        """! Constructs the DMCControllerState conformant with controller configuration provided by the user.

        @param controller_configuration DMC controller configuration
        @return controller_state DMC controller state
        """
        return DMCControllerState.zero_state(
            controller_configuration.D,
            controller_configuration.n_inputs,
            dtype=controller_configuration.dtype,
            device=controller_configuration.device,
        )


class DMCControllerClosedSystem(BaseDMCClosedSystem):
    """! Closed system with plant and Dynamic Matrix Control (DMC) controller.

    DMC controller solves the least squares problem to find the solution of the unconstrained optimisation problem and then casts the solution onto the feasible set.
    """

    def __init__(
        self,
        plant_dynamics: StateSpaceDynamics,
        controller_configuration: DMCControllerConfiguration,
        controller_state: DMCControllerState,
    ):
        """! Constructor of the DMCControllerClosedSystem class.

        @param plant_dynamics Plant dynamics
        @param controller_configuration DMC controller configuration
        @param controller_state DMC controller state
        """
        super().__init__(plant_dynamics, controller_configuration, controller_state)

    def _solve_constrained_problem(
        self,
        e_flat: torch.Tensor,
        du_flat: torch.Tensor,
    ) -> torch.Tensor:
        """! Solves a Karush-Kuhn-Tucker system with equality
        constraints corresponding to constraints on u or du
        violated by the least squares problem solution.

        @param e_flat Trajectory error tensor
            Shape: (N*n_outputs,)
        @param du_flat Unconstrained least squares problem solution
            Shape: (Nu*n_inputs,)
        @return du_flat Polished solution
            Shape: (Nu*n_inputs,)
        """
        du_flat_len = self.config.Nu * self.plant.n_inputs
        sum_input_deltas_array = self.config.sum_input_deltas_array

        current_u = self.state.current_u
        current_u_rep = current_u.repeat(self.config.N)
        u0 = self.config.operating_point.u
        u0_rep = u0.repeat(self.config.N)

        predicted_u = sum_input_deltas_array @ du_flat + current_u_rep

        eye_du = torch.eye(du_flat_len, dtype=self.dtype, device=self.device)

        constraint_rows = []
        constraint_targets = []

        # Build du constraints KKT blocks and targets
        if self.config.du_min is not None:
            mask = du_flat < self.config.du_min.repeat(self.config.Nu)
            if mask.any():
                constraint_rows.append(eye_du[mask])
                constraint_targets.append(
                    self.config.du_min[torch.where(mask)[0] % self.config.n_inputs]
                )
        if self.config.du_max is not None:
            mask = du_flat > self.config.du_max.repeat(self.config.Nu)
            if mask.any():
                constraint_rows.append(eye_du[mask])
                constraint_targets.append(
                    self.config.du_max[torch.where(mask)[0] % self.config.n_inputs]
                )

        # Build u constraints KKT blocks and targets
        if self.config.u_min is not None:
            u_min_rep = self.config.u_min.repeat(self.config.N)
            mask = predicted_u + u0_rep < u_min_rep
            if mask.any():
                u_min_shifted = self.config.u_min - u0 - current_u
                constraint_rows.append(sum_input_deltas_array[mask])
                constraint_targets.append(
                    u_min_shifted[torch.where(mask)[0] % self.config.n_inputs]
                )
        if self.config.u_max is not None:
            u_max_rep = self.config.u_max.repeat(self.config.N)
            mask = predicted_u + u0_rep > u_max_rep
            if mask.any():
                u_max_shifted = self.config.u_max - u0 - current_u
                constraint_rows.append(sum_input_deltas_array[mask])
                constraint_targets.append(
                    u_max_shifted[torch.where(mask)[0] % self.config.n_inputs]
                )

        if len(constraint_rows) == 0:
            return du_flat

        C = torch.cat(constraint_rows, dim=0)
        d = torch.cat(constraint_targets, dim=0)

        Lambda = self.config.regularisation * torch.eye(
            du_flat_len, dtype=self.dtype, device=self.device
        )
        H = self.config.M.T @ self.config.M + Lambda

        # Build the KKT matrix
        KKT = torch.cat(
            (
                torch.cat((H, C.T), dim=1),
                torch.cat(
                    (
                        C,
                        torch.zeros(
                            (C.shape[0], C.shape[0]),
                            dtype=self.dtype,
                            device=self.device,
                        ),
                    ),
                    dim=1,
                ),
            ),
            dim=0,
        )

        residual = torch.cat((self.config.M.T @ e_flat, d), dim=0)

        # We use pseudoinverse as KKT matrix may happen to be singular
        # because of the constraints redundancy
        solution = torch.linalg.pinv(KKT) @ residual

        return solution[:du_flat_len]

    def step(self, y: torch.Tensor, r_traj: torch.Tensor) -> torch.Tensor:
        """Compute next control input u given current measurement y and reference trajectory.

        @param y Current measured output
            Shape: (n_outputs,)
        @param r_traj Desired trajectory over prediction horizon
            Shape: (N, n_outputs)

        @return u Next control input
            Shape: (n_inputs,)
        """
        validate_tensor(y, "y", (self.plant.n_outputs,))
        if y.device != self.device:
            raise ValueError(
                f"`y` must be on the same device as controller ({self.device}), but it is on device {y.device}"
            )
        validate_tensor(r_traj, "r_traj", (self.config.N, self.plant.n_outputs))
        if r_traj.device != self.device:
            raise ValueError(
                f"`r_traj` must be on the same device as controller ({self.device}), but it is on device {r_traj.device}"
            )

        r_traj_flat = r_traj.reshape(-1)
        past_du_flat = self.state.past_du.reshape(-1)
        y_free_flat = y.repeat(self.config.N) + self.config.Mp @ past_du_flat
        e_flat = r_traj_flat - y_free_flat

        du_flat = self.config.K @ e_flat

        # Polishing (Active-set approach)
        if self.config.use_polishing:
            sum_input_deltas_array = self.config.sum_input_deltas_array
            current_u_rep = self.state.current_u.repeat(self.config.N)
            u0_rep = self.config.operating_point.u.repeat(self.config.N)
            predicted_u = sum_input_deltas_array @ du_flat + current_u_rep

            violated = False
            if self.config.du_min is not None:
                violated = violated or bool(
                    (du_flat < self.config.du_min.repeat(self.config.Nu)).any()
                )
            if self.config.du_max is not None:
                violated = violated or bool(
                    (du_flat > self.config.du_max.repeat(self.config.Nu)).any()
                )
            if self.config.u_min is not None:
                violated = violated or bool(
                    (
                        predicted_u + u0_rep < self.config.u_min.repeat(self.config.N)
                    ).any()
                )
            if self.config.u_max is not None:
                violated = violated or bool(
                    (
                        predicted_u + u0_rep > self.config.u_max.repeat(self.config.N)
                    ).any()
                )

            if violated:
                du_flat = self._solve_constrained_problem(e_flat, du_flat)

        # Calculated raw control increment
        du = du_flat[: self.plant.n_inputs]

        # Clamp control increment
        if self.config.du_min is not None:
            du = torch.clamp(du, min=self.config.du_min)
        if self.config.du_max is not None:
            du = torch.clamp(du, max=self.config.du_max)

        # Calculated raw control signal
        u_new = self.state.current_u + du

        # Clamp control signal
        if self.config.u_min is not None:
            u_new = torch.maximum(
                u_new, self.config.u_min - self.config.operating_point.u
            )
        if self.config.u_max is not None:
            u_new = torch.minimum(
                u_new, self.config.u_max - self.config.operating_point.u
            )

        # Actual control increment after enforcing constraints
        du_actual = u_new - self.state.current_u

        # Update controller state
        self.state.past_du = torch.cat(
            (du_actual.unsqueeze(0), self.state.past_du[:-1]), dim=0
        )
        self.state.current_u = u_new

        return u_new + self.config.operating_point.u

    def reset(self) -> None:
        """Reset the plant state and the controller state."""
        self.plant.reset()
        self.state = DMCControllerState.initial_state_for(self.config)
