import pytest
import torch

from akson import StateSpaceDynamics
from akson import (
    DMCControllerConfiguration,
    DMCControllerState,
    DMCControllerClosedSystem,
)


def _stable_siso_dynamics(**kwargs):
    # dx/dt = -x + u, y = x
    A = torch.tensor([[-1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    return StateSpaceDynamics.from_linear(A, B, C, D, **kwargs)


def _decoupled_mimo_dynamics(**kwargs):
    A = torch.tensor([[-1.0, 0.0], [0.0, -2.0]], dtype=torch.float64)
    B = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    return StateSpaceDynamics.from_linear(A, B, C, D, **kwargs)


def _siso_config(N=20, Nu=10, dt=0.5, duration=15.0, **config_kwargs):
    dynamics = _stable_siso_dynamics()
    t_step, S = dynamics.discrete_step_response(duration=duration, dt=dt)
    op = dynamics.operating_point
    config = DMCControllerConfiguration(S, N, Nu, op, **config_kwargs)
    state = DMCControllerState.initial_state_for(config)
    return dynamics, config, state


def _mimo_config(N=20, Nu=10, dt=0.5, duration=15.0, **config_kwargs):
    dynamics = _decoupled_mimo_dynamics()
    t_step, S = dynamics.discrete_step_response(duration=duration, dt=dt)
    op = dynamics.operating_point
    config = DMCControllerConfiguration(S, N, Nu, op, **config_kwargs)
    state = DMCControllerState.initial_state_for(config)
    return dynamics, config, state


def test_dmc_converges_to_constant_setpoint_siso():
    dynamics, config, state = _siso_config(regularisation=0.1)
    closed_system = DMCControllerClosedSystem(dynamics, config, state)

    r_traj = torch.tensor([[2.0]] * 60, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(
        r_traj, duration=30.0, dt=0.5, num_substeps=2
    )

    assert y_all[-1, 0].item() == pytest.approx(2.0, abs=0.1)


def test_dmc_tracks_setpoint_change():
    dynamics, config, state = _siso_config(regularisation=0.1)
    closed_system = DMCControllerClosedSystem(dynamics, config, state)

    r_traj = torch.tensor([[1.0]] * 30 + [[3.0]] * 30, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(
        r_traj, duration=30.0, dt=0.5, num_substeps=2
    )

    assert y_all[-1, 0].item() > 2.0


def test_dmc_mimo_tracks_both_channels():
    dynamics, config, state = _mimo_config(regularisation=0.1)
    closed_system = DMCControllerClosedSystem(dynamics, config, state)

    r_traj = torch.tensor([[1.5, -0.5]] * 60, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(
        r_traj, duration=30.0, dt=0.5, num_substeps=2
    )

    assert y_all[-1, 0].item() == pytest.approx(1.5, abs=0.15)
    assert y_all[-1, 1].item() == pytest.approx(-0.5, abs=0.15)


def test_dmc_respects_u_bounds_throughout_simulation():
    dynamics, config, state = _siso_config(
        regularisation=0.1,
        u_min=torch.tensor([-1.0]),
        u_max=torch.tensor([1.0]),
    )
    closed_system = DMCControllerClosedSystem(dynamics, config, state)

    r_traj = torch.tensor([[1000.0]] * 40, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(
        r_traj, duration=20.0, dt=0.5, num_substeps=2
    )

    assert (u_all[:, 0] <= 1.0 + 1e-6).all()
    assert (u_all[:, 0] >= -1.0 - 1e-6).all()


def test_dmc_respects_du_bounds_throughout_simulation():
    dynamics, config, state = _siso_config(
        regularisation=0.1,
        du_min=torch.tensor([-0.02]),
        du_max=torch.tensor([0.02]),
    )
    closed_system = DMCControllerClosedSystem(dynamics, config, state)

    r_traj = torch.tensor([[1000.0]] * 40, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(
        r_traj, duration=20.0, dt=0.5, num_substeps=2
    )

    du = u_all[1:, 0] - u_all[:-1, 0]
    assert (du <= 0.02 + 1e-9).all()
    assert (du >= -0.02 - 1e-9).all()


def test_dmc_with_polishing_still_respects_constraints():
    dynamics, config, state = _siso_config(
        regularisation=0.1,
        du_min=torch.tensor([-0.05]),
        du_max=torch.tensor([0.05]),
        u_min=torch.tensor([-1.0]),
        u_max=torch.tensor([1.0]),
        use_polishing=True,
    )
    closed_system = DMCControllerClosedSystem(dynamics, config, state)

    r_traj = torch.tensor([[1000.0]] * 40, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(
        r_traj, duration=20.0, dt=0.5, num_substeps=2
    )

    assert (u_all[:, 0] <= 1.0 + 1e-6).all()


def test_dmc_simulate_is_continuous_across_calls():
    dynamics, config, state = _siso_config(regularisation=0.1)
    closed_system = DMCControllerClosedSystem(dynamics, config, state)

    r_traj1 = torch.tensor([[1.0]] * 40, dtype=torch.float64)
    t1, y1, u1 = closed_system.simulate(r_traj1, duration=10.0, dt=0.5, num_substeps=1)

    r_traj2 = torch.tensor([[1.0]] * 40, dtype=torch.float64)
    t2, y2, u2 = closed_system.simulate(r_traj2, duration=10.0, dt=0.5, num_substeps=1)

    assert t2[0].item() == pytest.approx(t1[-1].item())
    assert y2[0, 0].item() == pytest.approx(y1[-1, 0].item(), abs=1e-8)


def test_dmc_controller_state_persists_across_calls():
    dynamics, config, state = _siso_config(regularisation=0.1)
    closed_system = DMCControllerClosedSystem(dynamics, config, state)

    r_traj = torch.tensor([[1.0]] * 20, dtype=torch.float64)
    closed_system.simulate(r_traj, duration=10.0, dt=0.5, num_substeps=1)

    # After some control action, current_u should have moved away from zero
    assert closed_system.state.current_u.abs().sum().item() > 0.0


def test_dmc_simulate_extends_short_r_traj_with_last_value():
    dynamics, config, state = _siso_config(regularisation=0.1)
    closed_system = DMCControllerClosedSystem(dynamics, config, state)

    # Provide far fewer reference points than num_steps + N requires
    r_traj = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(
        r_traj, duration=10.0, dt=0.5, num_substeps=1
    )

    assert t_all.shape[0] > 0
    assert y_all.shape[0] == t_all.shape[0]
