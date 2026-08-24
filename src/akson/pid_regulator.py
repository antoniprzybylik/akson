import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from .state_space_system import StateSpaceSystem, StateSpaceDynamics
import mini_ode
from ._utils import resolve_default_solver
from ._validation import (
    validate_tensor_shape_with_names,
    validate_tensor,
    validate_and_move_optional_tensor,
    validate_optional_tensors_le
)

@dataclass(frozen=True)
class PIDChannel:
    """! Single SISO channel of MIMO PID regulator. """
    output_idx: int
    input_idx: int
    K: float
    Ti: float
    Td: float
    
    def __post_init__(self):
        if not isinstance(self.output_idx, int) or self.output_idx < 0:
            raise ValueError("output_idx must be a nonnegative integer")
        if not isinstance(self.input_idx, int) or self.input_idx < 0:
            raise ValueError("input_idx must be a nonnegative integer")
        if self.Ti <= 0:
            raise ValueError("Ti must be positive")
        if self.Td < 0:
            raise ValueError("Td must be nonnegative")

class PIDRegulatorConfiguration:
    """! PID regulator configuration.

    The MIMO PID regulator consists of lattice of SISO PID
    regulators. Each of the SISO regulators works independently.
    """
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        channels: List[PIDChannel],
        t: float,
        u0: Optional[torch.Tensor] = None,
        u_min: Optional[torch.Tensor] = None,
        u_max: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu")
    ):
        """! Constructor of the PIDRegulatorConfiguration class

        @param n_inputs Number of plant inputs
        @param n_outputs Number of plant outputs
        @param channels SISO PID channels of MIMO PID regulator
        @param t Discretization constant
        @param u0 Base output (PID regulator output is relative to u0)
            Shape: (n_inputs,)
        @param u_min Minimum allowed control signal (absolute value, not u0 relative)
            Shape: (n_inputs,)
        @param u_max Maximum allowed control signal (absolute value, not u0 relative)
            Shape: (n_inputs,)
        @param dtype Torch dtype (default: torch.float64)
        @param device Torch device (default: torch.device("cpu"))
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # Channels validation
        io_pairs = set()
        for channel in channels:
            io_pair = (channel.input_idx, channel.output_idx)
            if n_inputs <= channel.input_idx:
                raise ValueError(f"Channel {io_pair} points to nonexisting input {channel.input_idx}.")
            if n_outputs <= channel.output_idx:
                raise ValueError(f"Channel {io_pair} points to nonexisting output {channel.output_idx}.")
            if io_pair in io_pairs:
                raise ValueError(f"Channel {io_pair} defined twice.")
            io_pairs.add(io_pair)

        self.dtype = dtype
        self.device = device

        # Validate and store bounds
        self.u_min = validate_and_move_optional_tensor(
                u_min, "u_min", (self.n_inputs,),
                desired_dtype=self.dtype, desired_device=self.device
        )
        self.u_max = validate_and_move_optional_tensor(
                u_max, "u_max", (self.n_inputs,),
                desired_dtype=self.dtype, desired_device=self.device
        )
        validate_optional_tensors_le(self.u_min, "u_min", self.u_max, "u_max")

        # Validate u0
        self.u0 = validate_and_move_optional_tensor(
                u0, "u0", (self.n_inputs,),
                desired_dtype=self.dtype, desired_device=self.device
        )
        validate_optional_tensors_le(self.u_min, "u_min", self.u0, "u0")
        validate_optional_tensors_le(self.u0, "u0", self.u_max, "u_max")

        # Build discrete PID coefficients table
        self.coeffs = torch.zeros((n_inputs, n_outputs, 3), dtype=self.dtype, device=self.device)
        for channel in channels:
            K = channel.K
            Ti = channel.Ti
            Td = channel.Td
            r0 = K * (1. + t / (2. * Ti) + Td / t)
            r1 = K * (t / (2. * Ti) - 2. * Td / t - 1.)
            r2 = K * Td / t
            self.coeffs[channel.input_idx, channel.output_idx, 0] = r0
            self.coeffs[channel.input_idx, channel.output_idx, 1] = r1
            self.coeffs[channel.input_idx, channel.output_idx, 2] = r2

        self.t = t


class PIDRegulatorState:
    """! PID regulator state. """
    def __init__(self,
        n_inputs: int,
        n_outputs: int,
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu")
    ):
        """! Constructor of the PIDRegulatorState class

        @param n_inputs Number of inputs to the system
        @param n_outputs Number of the system outputs
        @param dtype Torch dtype (default: torch.float64)
        @param device Torch device (default: torch.device("cpu"))
        """
        self.dtype = dtype
        self.device = device

        self.e_prev = torch.zeros((n_outputs,), dtype=self.dtype, device=self.device)
        self.e_prev_prev = torch.zeros((n_outputs,), dtype=self.dtype, device=self.device)
        self.u_prev = torch.zeros((n_inputs,), dtype=self.dtype, device=self.device)

    @staticmethod
    def initial_state_for(regulator_configuration: 'PIDRegulatorConfiguration') -> 'PIDRegulatorState':
        """! Constructs DMC regulator zero state compatible with provided regulator configuration.

        @param regulator_configuration PID regulator configuration
        @return regulator_state PID regulator state
        """
        return PIDRegulatorState(
            regulator_configuration.n_inputs,
            regulator_configuration.n_outputs,
            dtype=regulator_configuration.dtype,
            device=regulator_configuration.device,
        )

class PIDRegulatorClosedSystem:
    """! Closed system with plant and PID regulator. """
    def __init__(self,
        plant_dynamics: 'StateSpaceDynamics',
        initial_state: torch.Tensor,
        config: 'PIDRegulatorConfiguration',
        state: 'PIDRegulatorState'
    ):
        """! Constructor of the PIDRegulatorClosedSystem class.

        @param plant_dynamics Plant dynamics
        @param initial_state Initial plant state
        @param config PID regulator configuration
        @param state PID regulator state
        """
        if plant_dynamics.n_inputs != config.n_inputs:
            raise ValueError("Plant dynamics and PID configuration do not conform. Different assumed number of system inputs.")
        if plant_dynamics.n_outputs != config.n_outputs:
            raise ValueError("Plant dynamics and PID configuration do not conform. Different assumed number of system outputs.")

        self.plant = StateSpaceSystem(plant_dynamics, x=initial_state)
        self.config = config
        self.state = state

        self.dtype = self.config.dtype
        self.device = self.config.device

    def step(self, y: torch.Tensor, setpoint: torch.Tensor) -> torch.Tensor:
        """! Compute next control input u given current measurement y and desired output "setpoint".
    
        @param y Current measured output
            Shape: (n_outputs,)
        @param setpoint Desired plant output
            Shape: (n_outputs,)
    
        @return u Next control input
            Shape: (n_inputs,)
        """
        # Validate current measured output y
        validate_tensor(y, "y", (self.plant.n_outputs,))
        if y.device != self.device:
            raise ValueError(f"`y` must be on the same device as regulator ({self.device}), but it is on device {y.device}")

        # Validate setpoint
        validate_tensor(setpoint, "setpoint", (self.plant.n_outputs,))
        if setpoint.device != self.device:
            raise ValueError(f"`setpoint` must be on the same device as regulator ({self.device}), but it is on device {setpoint.device}")
        
        e = setpoint - y
        u_new = self.config.coeffs[:, :, 0] @ e + self.config.coeffs[:, :, 1] @ self.state.e_prev + self.config.coeffs[:, :, 2] @ self.state.e_prev_prev + self.state.u_prev

        # Clamping anti-windup
        if self.config.u_min is not None:
            violated = (u_new < (self.config.u_min-self.config.u0))
            u_new[violated] = (self.config.u_min-self.config.u0)[violated]
        if self.config.u_max is not None:
            violated = (u_new > (self.config.u_max-self.config.u0))
            u_new[violated] = (self.config.u_max-self.config.u0)[violated]

        self.state.e_prev_prev = self.state.e_prev
        self.state.e_prev = e
        self.state.u_prev = u_new

        return self.config.u0 + u_new

    def simulate(
        self,
        setpoints: torch.Tensor,
        duration: float,
        num_substeps: Optional[int] = None,
        solver: Optional[object] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """! Simulate a closed-loop system.
    
        @param setpoints Setpoints for consecutive regulator steps
            Shape: (length, n_outputs). If shorter than needed, extended with last element
        @param duration Total simulation time
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
            setpoints, "setpoints",
            (None, self.plant.n_outputs), ("n_setpoints", None),
            ("number of setpoints", None)
        )
    
        # Solver step size
        if num_substeps is not None:
            step_size = self.config.t / num_substeps
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
        num_steps = int(duration / self.config.t)

        # Extend / cut the setpoints tensor
        setpoints_needed = num_steps
        stp_full = setpoints.to(dtype=self.dtype, device=self.device)
        if stp_full.shape[0] < setpoints_needed:
            pad_len = setpoints_needed - stp_full.shape[0]
            last = stp_full[-1:].repeat(pad_len, 1)
            stp_full = torch.cat([stp_full, last], dim=0)
        else:
            stp_full = stp_full[:setpoints_needed]

        t_all = []
        y_all = []
        u_all = []
        current_x = self.plant.x.clone()
        current_t = self.plant.simulation_time
        u = self.config.u0.clone()
        self.plant.dynamics._validate_u(u)
        y = self.plant.dynamics._g(current_x, u)
        t_all.append(current_t)
        y_all.append(y)
        u_all.append(u)
        for i in range(num_steps):
            # Get setpoint for this step
            stp = stp_full[i]

            # Calculate control input from the regulator
            u_new = self.step(y, stp)
            self.plant.dynamics._validate_u(u_new)

            # Simulate the system on current timespan
            traced_ode_fn = self.plant.dynamics._create_traced_ode_function(u_new)
            t_span = (current_t, current_t + self.config.t)
            y0 = current_x.to(torch.float64)
            t_sub, x_sub = solver.solve(traced_ode_fn, t_span, y0) 
            self.plant.dynamics._validate_x(x_sub)

            # Compute y at each sub-point (validated against y_min/y_max)
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

    def reset(
        self
    ) -> None:
        """Reset the plant state and the regulator state. """
        self.plant.reset()
        self.state = PIDRegulatorState.initial_state_for(self.config)
