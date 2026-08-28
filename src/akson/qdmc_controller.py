import torch
import numpy as np
import cvxpy as cp
import mini_ode
from copy import copy
from typing import Optional, Literal, Tuple
from .state_space_system import StateSpaceSystem, StateSpaceDynamics, OperatingPoint
from ._utils import resolve_default_solver
from ._dmc_common import (
    build_dynamic_matrices,
    build_sum_input_deltas_array,
    zero_past_du_and_current_u,
    BaseDMCState,
    BaseDMCClosedSystem
)
from ._validation import (
    validate_tensor_shape_with_names,
    validate_tensor,
    validate_and_move_optional_tensor,
    validate_optional_tensors_le
)


class QDMCControllerConfiguration:
    """! Quadratic Dynamic Matrix Control (QDMC) controller configuration. """
    def __init__(
        self,
        step_response: torch.Tensor,
        N: int,
        Nu: int,
        operating_point: OperatingPoint,
        regularisation: float = 0.0,
        du_min: Optional[float] = None,
        du_max: Optional[float] = None,
        u_min: Optional[torch.Tensor] = None,
        u_max: Optional[torch.Tensor] = None,
        y_min: Optional[torch.Tensor] = None,
        y_max: Optional[torch.Tensor] = None,
        policy: Literal["strict", "soft", "minimize_violation"] = "strict",
        rho_min: Optional[float] = None,
        rho_max: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """! Constructor of the QDMCControllerConfiguration class

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
        @param y_min Minimal allowed predicted plant output
            Shape: (n_outputs,)
        @param y_max Maximum allowed predicted plant output
            Shape: (n_outputs,)
        @param policy Infeasibility handling policy
        @param rho_min Penalty for control signal lower bound constraint violation (does not apply to "strict" policy)
        @param rho_max Penalty for control signal upper bound constraint violation (does not apply to "strict" policy)
        @param dtype Torch dtype
        @param device Torch device
        """
        # Validation
        validate_tensor_shape_with_names(
            step_response, "step_response",
            (None, None, None), ("length", "n_outputs", "n_inputs"),
            ("length of step response", "number of system outputs", "number of system inputs")
        )
        if step_response.shape[0] < 1:
            raise ValueError("step_response must not contain less than one sample")
        if step_response.shape[1] < 1:
            raise ValueError(f"System output size must not be smaller than one. Got shape {step_response.shape}")
        if step_response.shape[2] < 1:
            raise ValueError(f"System input size must not be smaller than one. Got shape {step_response.shape}")
        if N < 1:
            raise ValueError("Prediction horizon N must be a positive integer")
        if Nu < 1:
            raise ValueError("Control horizon Nu must be a positive integer")
        if regularisation < 0:
            raise ValueError("Regularisation factor must be nonnegative")
        if rho_min is not None and rho_min < 0:
            raise ValueError("rho_min must be nonnegative")
        if rho_max is not None and rho_max < 0:
            raise ValueError("rho_max must be nonnegative")
        if policy not in ("strict", "soft", "minimize_violation"):
            raise ValueError(f"Invalid policy: \"{policy}\"")
        if policy == "strict" and (rho_min is not None or rho_max is not None):
            raise ValueError(f"Parameters rho_min and rho_max are invalid for policy \"{policy}\"")
        if policy in ("soft", "minimize_violation") and (rho_min is None or rho_max is None):
            raise ValueError(f"Parameters rho_min and rho_max are required for policy \"{policy}\"")
        
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
        self.policy = policy
        self.rho_min = rho_min
        self.rho_max = rho_max

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
            dtype=self.dtype
        )

        # Validate and store bounds
        self.du_min = validate_and_move_optional_tensor(
                du_min, "du_min", (self.n_inputs,),
                desired_dtype=self.dtype, desired_device=self.device
        )
        self.du_max = validate_and_move_optional_tensor(
                du_max, "du_max", (self.n_inputs,),
                desired_dtype=self.dtype, desired_device=self.device
        )
        validate_optional_tensors_le(self.du_min, "du_min", self.du_max, "du_max")
        self.u_min = validate_and_move_optional_tensor(
                u_min, "u_min", (self.n_inputs,),
                desired_dtype=self.dtype, desired_device=self.device
        )
        self.u_max = validate_and_move_optional_tensor(
                u_max, "u_max", (self.n_inputs,),
                desired_dtype=self.dtype, desired_device=self.device
        )
        validate_optional_tensors_le(self.u_min, "u_min", self.u_max, "u_max")
        self.y_min = validate_and_move_optional_tensor(
                y_min, "y_min", (self.n_outputs,),
                desired_dtype=self.dtype, desired_device=self.device
        )
        self.y_max = validate_and_move_optional_tensor(
                y_max, "y_max", (self.n_outputs,),
                desired_dtype=self.dtype, desired_device=self.device
        )
        validate_optional_tensors_le(self.y_min, "y_min", self.y_max, "y_max")

        # Build and store dynamic matrices
        self.M, self.Mp = build_dynamic_matrices(self.S, self.N, self.Nu)

        # Build and store "sum input deltas" array
        self.sum_input_deltas_array = build_sum_input_deltas_array(self.N, self.Nu, self.n_inputs, self.dtype, self.device)


class QDMCControllerState(BaseDMCState):
    """! QDMC controller state.

    Beside standard DMC state, we store here the CVXPY problem data.
    """
    class _ProblemData:
        """! CVXPY problem data. """
        def __init__(self, config: QDMCControllerConfiguration):
            """! Constructs problem data from QDMC controller configuration. """
            # Helper arrays for building CVXPY problem
            M_np = config.M.cpu().numpy()
            sum_input_deltas_array_np = config.sum_input_deltas_array.cpu().numpy()
            u0 = config.operating_point.u
            u0_rep_np = np.tile(u0.cpu().numpy(), config.N)
            y0 = config.operating_point.y
            y0_rep_np = np.tile(y0.cpu().numpy(), config.N)
            du_min_rep_np = np.tile(config.du_min.cpu().numpy(), config.Nu) if config.du_min is not None else None
            du_max_rep_np = np.tile(config.du_max.cpu().numpy(), config.Nu) if config.du_max is not None else None
            u_min_rep_np = np.tile(config.u_min.cpu().numpy(), config.N) if config.u_min is not None else None
            u_max_rep_np = np.tile(config.u_max.cpu().numpy(), config.N) if config.u_max is not None else None
            y_min_rep_np = np.tile(config.y_min.cpu().numpy(), config.N) if config.y_min is not None else None
            y_max_rep_np = np.tile(config.y_max.cpu().numpy(), config.N) if config.y_max is not None else None
    
            # CVXPY problem parameters
            self.u_prev_param = cp.Parameter(config.n_inputs)
            self.u_prev_param.value = np.zeros(config.n_inputs)
            self.free_traj_param = cp.Parameter(config.N * config.n_outputs)
            self.target_param = cp.Parameter(config.N * config.n_outputs)
    
            # CVXPY problem variable
            self.x = cp.Variable(config.Nu * config.n_inputs)
    
            # Collect CVXPY problem constraints in this array
            constraints = []
    
            # Control signal increment constraints
            if config.du_min is not None:
                constraints.append(du_min_rep_np <= self.x)
            if config.du_max is not None:
                constraints.append(du_max_rep_np >= self.x)
    
            # Control signal constraints
            if u_min_rep_np is not None:
                constraints.append(
                    cp.hstack([self.u_prev_param for _ in range(config.N)])
                    + sum_input_deltas_array_np @ self.x
                    >= u_min_rep_np - u0_rep_np
                )
            if u_max_rep_np is not None:
                constraints.append(
                    cp.hstack([self.u_prev_param for _ in range(config.N)])
                    + sum_input_deltas_array_np @ self.x
                    <= u_max_rep_np - u0_rep_np
                )
    
            strict_constraints = list(constraints)
            soft_constraints = list(constraints)
    
            # Predicted plant output constraints
            if y_min_rep_np is not None:
                strict_constraints.append(
                    y_min_rep_np - y0_rep_np <= self.free_traj_param + M_np @ self.x
                )
                self.epsilon_min = cp.Variable(config.N * config.n_outputs)
                soft_constraints.append(self.epsilon_min >= 0.0)
                soft_constraints.append(
                    y_min_rep_np - y0_rep_np - self.epsilon_min <= self.free_traj_param + M_np @ self.x
                )
            if y_max_rep_np is not None:
                strict_constraints.append(
                    y_max_rep_np - y0_rep_np >= self.free_traj_param + M_np @ self.x
                )
                self.epsilon_max = cp.Variable(config.N * config.n_outputs)
                soft_constraints.append(self.epsilon_max >= 0.0)
                soft_constraints.append(
                    y_max_rep_np - y0_rep_np + self.epsilon_max >= self.free_traj_param + M_np @ self.x
                )
    
            base_objective = (
                cp.sum_squares(M_np @ self.x - self.target_param)
                + config.regularisation * cp.sum_squares(self.x)
            )
    
            # Construct CVXPY problem and fallback problem depending on the policy
            if config.policy == "strict":
                self.problem = cp.Problem(cp.Minimize(base_objective), strict_constraints)
                self.fallback_problem = None
            elif config.policy == "soft":
                objective = (
                    base_objective
                    + config.rho_min * cp.sum_squares(self.epsilon_min)
                    + config.rho_max * cp.sum_squares(self.epsilon_max)
                )
                self.problem = cp.Problem(cp.Minimize(objective), soft_constraints)
                self.fallback_problem = None
            elif config.policy == "minimize_violation":
                self.problem = cp.Problem(cp.Minimize(base_objective), strict_constraints)
                fallback_objective = (
                    config.rho_min * cp.sum_squares(self.epsilon_min)
                    + config.rho_max * cp.sum_squares(self.epsilon_max)
                )
                self.fallback_problem = cp.Problem(cp.Minimize(fallback_objective), soft_constraints)

    def __init__(
        self,
        past_du: torch.Tensor,
        current_u: torch.Tensor,
        config: QDMCControllerConfiguration,
        warm_start_x: Optional[torch.Tensor] = None,
    ):
        super().__init__(past_du, current_u, dtype=config.dtype, device=config.device)

        self.warm_start_x = validate_and_move_optional_tensor(
                warm_start_x, "warm_start_x", (config.Nu*config.n_inputs,),
                desired_dtype=config.dtype, desired_device=config.device
        )
        self._problem_data = self._ProblemData(config)

    @staticmethod
    def initial_state_for(
        controller_configuration: QDMCControllerConfiguration,
    ) -> QDMCControllerState:
        """! Builds initial state conforming with given QDMC controller configuration.

        @param controller_configuration QDMC controller configuration
        @return controller_state QDMC controller state
        """
        past_du, current_u = zero_past_du_and_current_u(
            controller_configuration.D, controller_configuration.n_inputs,
            controller_configuration.dtype, controller_configuration.device
        )
        return QDMCControllerState(
            past_du, current_u,
            controller_configuration, warm_start_x=None,
        )


class QDMCControllerClosedSystem(BaseDMCClosedSystem):
    """! Closed system with plant and Quadratic Dynamic Matrix Control (QDMC) controller.

    In each step, QDMC finds the exact solution of the quadratic programming
    problem. This is different from heurisitic approach of classical DMC.
    """
    def __init__(
        self,
        plant_dynamics: StateSpaceDynamics,
        controller_configuration: QDMCControllerConfiguration,
        controller_state: QDMCControllerState
    ):
        """! Constructor of the QDMCControllerClosedSystem class.

        @param plant_dynamics Plant dynamics
        @param controller_configuration QDMC controller configuration
        @param controller_state QDMC controller state
        """
        super().__init__(plant_dynamics, controller_configuration, controller_state)

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
            raise ValueError(f"`y` must be on the same device as controller ({self.device}), but it is on device {y.device}")
        validate_tensor(r_traj, "r_traj", (self.config.N, self.plant.n_outputs))
        if r_traj.device != self.device:
            raise ValueError(f"`r_traj` must be on the same device as controller ({self.device}), but it is on device {r_traj.device}")
    
        r_traj_flat = r_traj.reshape(-1)
        past_du_flat = self.state.past_du.reshape(-1)
        y_free_flat = y.repeat(self.config.N) + self.config.Mp @ past_du_flat
        e_flat = r_traj_flat - y_free_flat
        
        y0 = self.config.operating_point.y
        y0_rep = torch.tile(y0, (self.config.N,))

        # Set CVXPY problem parameters values
        self.state._problem_data.free_traj_param.value = (y_free_flat - y0_rep).cpu().numpy()
        self.state._problem_data.target_param.value = e_flat.cpu().numpy()
        self.state._problem_data.u_prev_param.value = self.state.current_u.cpu().numpy()

        # Warm start. We hope that the current step problem solution is
        # close to the previous step problem solution
        if self.state.warm_start_x is not None:
            self.state._problem_data.x.value = self.state.warm_start_x.cpu().numpy()

        # Use Splitting Conic Solver - a QP solver supporting warm start
        self.state._problem_data.problem.solve(solver=cp.SCS, warm_start=True)
        if self.state._problem_data.x.value is None and self.state._problem_data.fallback_problem is not None:
            self.state._problem_data.fallback_problem.solve(solver=cp.SCS, warm_start=True)

        if self.state._problem_data.x.value is None:
            if self.config.policy == "strict":
                raise RuntimeError("QP solver failed in QDMC step. There is no feasible plant input. The current QDMC policy is \"strict\". Try loosening policy or constraints.")
            else:
                raise RuntimeError("QP solver failed in QDMC step. There is no feasible plant input. Try loosening constraints.")

        du_flat = torch.tensor(self.state._problem_data.x.value, dtype=self.dtype, device=self.device)
        du = du_flat[:self.plant.n_inputs]

        # Store shifted present step solution as the warm start for next step
        shifted = torch.cat([du_flat[self.plant.n_inputs:], du_flat[:self.plant.n_inputs]], dim=0)

        # Polishing: Clamp control increment
        # (The QP solver solution may still violate constraints by very small ammount)
        if self.config.du_min is not None:
            du = torch.clamp(du, min=self.config.du_min)
        if self.config.du_max is not None:
            du = torch.clamp(du, max=self.config.du_max)
    
        # Calculated raw control signal
        u_new = self.state.current_u + du
    
        # Polishing: Clamp control signal
        # (The QP solver solution may still violate constraints by very small ammount)
        if self.config.u_min is not None:
            u_new = torch.maximum(u_new, self.config.u_min - self.config.operating_point.u)
        if self.config.u_max is not None:
            u_new = torch.minimum(u_new, self.config.u_max - self.config.operating_point.u)
    
        # Actual control increment after polishing
        du_actual = u_new - self.state.current_u

        # Update the state
        self.state.past_du = torch.cat((du_actual.unsqueeze(dim=0), self.state.past_du[:-1]), dim=0)
        self.state.current_u = u_new
        self.state.warm_start_x = shifted

        return self.config.operating_point.u + u_new

    def reset(
        self
    ) -> None:
        """Reset the plant state and the controller state. """
        self.plant.reset()
        self.state = QDMCControllerState.initial_state_for(self.config)
