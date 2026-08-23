import torch
import mini_ode
import warnings
from typing import Optional, Tuple
from .state_space_system import StateSpaceSystem, StateSpaceDynamics, OperatingPoint
from ._utils import resolve_default_solver
from ._dmc_common import (
    build_dynamic_matrices,
    build_sum_input_deltas_array,
    zero_past_du_and_current_u,
    BaseDMCState,
)
from ._validation import (
    validate_tensor_shape_with_names,
    validate_tensor,
    validate_and_move_optional_tensor,
    validate_optional_tensors_le
)

class DMCRegulatorConfiguration:
    """! Dynamic Matrix Control (DMC) regulator configuration. """
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
        device: Optional[torch.device] = None
    ):
        """! Constructor of the DMCRegulatorConfiguration class

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
        validate_tensor(operating_point.y, "operating point output", (self.n_inputs,))
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

        # Build and store dynamic matrices
        self.M, self.Mp = build_dynamic_matrices(self.S, self.N, self.Nu)

        # Compute K = (M^T*M + lambda*I)^-1 M^T
        MTM = self.M.T @ self.M
        Lambda = (
            self.regularisation *
            torch.eye(
                self.Nu * self.n_inputs,
                dtype=self.dtype,
                device=self.device
            )
        )
        self.K = torch.linalg.solve(MTM + Lambda, self.M.T)

        # Build and store "sum input deltas" array
        self.sum_input_deltas_array = build_sum_input_deltas_array(self.N, self.Nu, self.n_inputs, self.dtype, self.device)

class DMCRegulatorState(BaseDMCState):
    @staticmethod
    def zero_state(
        dynamics_horizon: int,
        n_inputs: int,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu")
    ) -> DMCRegulatorState:
        """! Constructs zero state for the DMC regulator.

        @param dynamics_horizon Dynamics horizon length
        @param n_inputs Number of inputs to the system
        @return regulator_state DMC regulator state
        """
        past_du, current_u = zero_past_du_and_current_u(dynamics_horizon, n_inputs, dtype, device)
        return DMCRegulatorState(past_du, current_u, dtype=dtype, device=device)

    @staticmethod
    def initial_state_for(
        regulator_configuration: DMCRegulatorConfiguration
    ) -> DMCRegulatorState:
        """! Constructs the DMCRegulatorState conformant with regulator configuration provided by the user.

        @param regulator_configuration DMC regulator configuration
        @return regulator_state DMC regulator state
        """
        return DMCRegulatorState.zero_state(
            regulator_configuration.D,
            regulator_configuration.n_inputs,
            dtype=regulator_configuration.dtype,
            device=regulator_configuration.device
        )

class DMCRegulatorClosedSystem:
    """! Closed system with plant and Dynamic Matrix Control (DMC) regulator.

    DMC regulator solves the least squares problem to find the solution of the unconstrained optimisation problem and then casts the solution onto the feasible set.
    """
    def __init__(
        self,
        plant_dynamics: 'StateSpaceDynamics',
        regulator_configuration: 'DMCRegulatorConfiguration',
        regulator_state: 'DMCRegulatorState'
    ):
        """! Constructor of the DMCRegulatorClosedSystem class.

        @param plant_dynamics Plant dynamics
        @param regulator_configuration DMC regulator configuration
        @param regulator_state DMC regulator state
        """
        if regulator_state.past_du.shape[0] != regulator_configuration.D-1:
            raise ValueError("Regulator state and regulator configuration do not conform. Different assumed dynamics horizon.")
        if regulator_state.past_du.shape[1] != regulator_configuration.n_inputs:
            raise ValueError("Regulator state and regulator configuration do not conform. Different assumed number of system inputs.")
        if regulator_state.current_u.shape[0] != regulator_configuration.n_inputs:
            raise ValueError("Regulator state and regulator configuration do not conform. Different assumed number of system inputs.")

        if plant_dynamics.n_inputs != regulator_configuration.n_inputs:
            raise ValueError("Plant dynamics and regulator configuration do not conform. Different assumed number of system inputs.")
        if plant_dynamics.n_outputs != regulator_configuration.n_outputs:
            raise ValueError("Plant dynamics and regulator configuration do not conform. Different assumed number of system outputs.")
        validate_tensor(
            regulator_configuration.operating_point.x,
            "regulator_configuration.operating_point.x",
            (plant_dynamics.state_size,)
        )

        if plant_dynamics.device != regulator_configuration.device or regulator_configuration.device != regulator_state.device:
            raise ValueError("Devices do not match.")
        self.device = plant_dynamics.device
        if plant_dynamics.dtype != regulator_configuration.dtype or regulator_configuration.dtype != regulator_state.dtype:
            raise ValueError("Datatypes do not match.")
        self.dtype = plant_dynamics.dtype

        if (plant_dynamics.u_min is not None and regulator_configuration.u_min is not None and (plant_dynamics.u_min > regulator_configuration.u_max).any()) or (plant_dynamics.u_min is not None and regulator_configuration.u_min is None):
            warnings.warn(
                "Constraints on u_min are looser in the regulator than in the plant. This may lead to regulator feeding infeasible input into the plant",
                RuntimeWarning,
            )

        if (plant_dynamics.u_max is not None and regulator_configuration.u_max is not None and (plant_dynamics.u_max > regulator_configuration.u_max).any()) or (plant_dynamics.u_max is not None and regulator_configuration.u_max is None):
            warnings.warn(
                "Constraints on u_max are looser in the regulator than in the plant. This may lead to regulator feeding infeasible input into the plant",
                RuntimeWarning,
            )

        self.plant = StateSpaceSystem(plant_dynamics, x=regulator_configuration.operating_point.x)
        self.config = regulator_configuration
        self.state = regulator_state

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
        KKT = torch.cat((
            torch.cat((H, C.T), dim=1),
            torch.cat((C, torch.zeros((C.shape[0], C.shape[0]), dtype=self.dtype, device=self.device)), dim=1),
        ), dim=0)
    
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
            raise ValueError(f"`y` must be on the same device as regulator ({self.device}), but it is on device {y.device}")
        validate_tensor(r_traj, "r_traj", (self.config.N, self.plant.n_outputs))
        if r_traj.device != self.device:
            raise ValueError(f"`r_traj` must be on the same device as regulator ({self.device}), but it is on device {r_traj.device}")
    
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
                violated = violated or bool((du_flat < self.config.du_min.repeat(self.config.Nu)).any())
            if self.config.du_max is not None:
                violated = violated or bool((du_flat > self.config.du_max.repeat(self.config.Nu)).any())
            if self.config.u_min is not None:
                violated = violated or bool((predicted_u + u0_rep < self.config.u_min.repeat(self.config.N)).any())
            if self.config.u_max is not None:
                violated = violated or bool((predicted_u + u0_rep > self.config.u_max.repeat(self.config.N)).any())
    
            if violated:
                du_flat = self._solve_constrained_problem(e_flat, du_flat)
    
        # Calculated raw control increment
        du = du_flat[:self.plant.n_inputs]
    
        # Clamp control increment
        if self.config.du_min is not None:
            du = torch.clamp(du, min=self.config.du_min)
        if self.config.du_max is not None:
            du = torch.clamp(du, max=self.config.du_max)
    
        # Calculated raw control signal
        u_new = self.state.current_u + du
    
        # Clamp control signal
        if self.config.u_min is not None:
            u_new = torch.maximum(u_new, self.config.u_min - self.config.operating_point.u)
        if self.config.u_max is not None:
            u_new = torch.minimum(u_new, self.config.u_max - self.config.operating_point.u)
    
        # Actual control increment after enforcing constraints
        du_actual = u_new - self.state.current_u
    
        # Update regulator state
        self.state.past_du = torch.cat((du_actual.unsqueeze(0), self.state.past_du[:-1]), dim=0)
        self.state.current_u = u_new
    
        return u_new + self.config.operating_point.u

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
        @param dt Time step for regulator updates
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
            r_traj, "r_traj",
            (None, self.plant.n_outputs), ("length", None),
            ("length of the reference trajectory", None)
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
    
        # Number of regulator control computation steps
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
            r_traj_step = r_full[i+1 : i+1 + self.config.N]

            # Calculate control input from the regulator
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
    
        return torch.tensor(t_all, dtype=self.dtype, device=self.device), torch.stack(y_all), torch.stack(u_all)
