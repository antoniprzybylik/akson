import pytest
import torch

from akson import StateSpaceDynamics
from akson import (
    PIDChannel,
    PIDControllerConfiguration,
    PIDControllerState,
    PIDControllerClosedSystem,
)


def _stable_siso_dynamics(**kwargs):
    # dx/dt = -x + u, y = x
    A = torch.tensor([[-1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    return StateSpaceDynamics.from_linear(A, B, C, D, **kwargs)


def _decoupled_mimo_dynamics(**kwargs):
    # Two independent first-order channels: dx1/dt = -x1 + u1, dx2/dt = -2 x2 + u2
    A = torch.tensor([[-1.0, 0.0], [0.0, -2.0]], dtype=torch.float64)
    B = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    return StateSpaceDynamics.from_linear(A, B, C, D, **kwargs)


def _make_siso_pid(K=2.0, Ti=1.0, Td=0.0, t=0.1, **config_kwargs):
    channel = PIDChannel(output_idx=0, input_idx=0, K=K, Ti=Ti, Td=Td)
    config = PIDControllerConfiguration(
        n_inputs=1, n_outputs=1, channels=[channel], t=t,
        u0=torch.tensor([0.0], dtype=torch.float64),
        **config_kwargs,
    )
    state = PIDControllerState.initial_state_for(config)
    return config, state


def test_pid_converges_to_constant_setpoint_siso():
    config, state = _make_siso_pid(K=2.0, Ti=1.0)
    dynamics = _stable_siso_dynamics()
    closed_system = PIDControllerClosedSystem(
        dynamics, torch.tensor([0.0], dtype=torch.float64), config, state
    )
    setpoints = torch.tensor([[2.0]] * 200, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(setpoints, duration=20.0, num_substeps=2)

    assert y_all[-1, 0].item() == pytest.approx(2.0, abs=0.05)


def test_pid_tracks_setpoint_change():
    config, state = _make_siso_pid(K=2.0, Ti=1.0)
    dynamics = _stable_siso_dynamics()
    closed_system = PIDControllerClosedSystem(
        dynamics, torch.tensor([0.0], dtype=torch.float64), config, state
    )
    setpoints = torch.tensor([[1.0]] * 100 + [[3.0]] * 100, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(setpoints, duration=20.0, num_substeps=2)

    # By the end of the run the output should have moved towards the second
    # setpoint (3.0) rather than stayed near the first (1.0)
    assert y_all[-1, 0].item() > 2.0


def test_pid_mimo_channels_track_independently():
    channel0 = PIDChannel(output_idx=0, input_idx=0, K=2.0, Ti=1.0, Td=0.0)
    channel1 = PIDChannel(output_idx=1, input_idx=1, K=3.0, Ti=1.0, Td=0.0)
    config = PIDControllerConfiguration(
        n_inputs=2, n_outputs=2, channels=[channel0, channel1], t=0.1,
        u0=torch.tensor([0.0, 0.0], dtype=torch.float64),
    )
    state = PIDControllerState.initial_state_for(config)
    dynamics = _decoupled_mimo_dynamics()
    closed_system = PIDControllerClosedSystem(
        dynamics, torch.tensor([0.0, 0.0], dtype=torch.float64), config, state
    )
    setpoints = torch.tensor([[1.0, -1.0]] * 200, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(setpoints, duration=20.0, num_substeps=2)

    assert y_all[-1, 0].item() == pytest.approx(1.0, abs=0.05)
    assert y_all[-1, 1].item() == pytest.approx(-1.0, abs=0.05)


def test_pid_respects_u_bounds_throughout_simulation():
    config, state = _make_siso_pid(
        K=50.0, Ti=1.0,
        u_min=torch.tensor([-1.0]), u_max=torch.tensor([1.0]),
    )
    dynamics = _stable_siso_dynamics()
    closed_system = PIDControllerClosedSystem(
        dynamics, torch.tensor([0.0], dtype=torch.float64), config, state
    )
    # Setpoint far outside the plant's reasonable range forces the
    # controller to saturate for a long time.
    setpoints = torch.tensor([[1000.0]] * 100, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(setpoints, duration=10.0, num_substeps=2)

    assert (u_all[:, 0] <= 1.0 + 1e-9).all()
    assert (u_all[:, 0] >= -1.0 - 1e-9).all()


def test_pid_simulate_is_continuous_across_calls():
    config, state = _make_siso_pid(K=1.0, Ti=2.0)
    dynamics = _stable_siso_dynamics()
    closed_system = PIDControllerClosedSystem(
        dynamics, torch.tensor([0.0], dtype=torch.float64), config, state
    )
    setpoints1 = torch.tensor([[1.0]] * 50, dtype=torch.float64)
    t1, y1, u1 = closed_system.simulate(setpoints1, duration=5.0, num_substeps=1)

    setpoints2 = torch.tensor([[1.0]] * 50, dtype=torch.float64)
    t2, y2, u2 = closed_system.simulate(setpoints2, duration=5.0, num_substeps=1)

    # The second run's time should pick up exactly where the first left off,
    # with no jump in the output at the boundary.
    assert t2[0].item() == pytest.approx(t1[-1].item())
    assert y2[0, 0].item() == pytest.approx(y1[-1, 0].item(), abs=1e-8)


def test_pid_closed_system_reflects_plant_state_between_calls():
    config, state = _make_siso_pid(K=1.0, Ti=2.0)
    dynamics = _stable_siso_dynamics()
    closed_system = PIDControllerClosedSystem(
        dynamics, torch.tensor([0.0], dtype=torch.float64), config, state
    )
    setpoints = torch.tensor([[1.0]] * 50, dtype=torch.float64)
    closed_system.simulate(setpoints, duration=5.0, num_substeps=1)

    assert closed_system.plant.simulation_time == pytest.approx(5.0)
