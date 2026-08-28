import pytest
import torch

from akson import StateSpaceDynamics
from akson import (
    QDMCControllerConfiguration,
    QDMCControllerState,
    QDMCControllerClosedSystem,
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
    config = QDMCControllerConfiguration(S, N, Nu, op, **config_kwargs)
    state = QDMCControllerState.initial_state_for(config)
    return dynamics, config, state


def _mimo_config(N=20, Nu=10, dt=0.5, duration=15.0, **config_kwargs):
    dynamics = _decoupled_mimo_dynamics()
    t_step, S = dynamics.discrete_step_response(duration=duration, dt=dt)
    op = dynamics.operating_point
    config = QDMCControllerConfiguration(S, N, Nu, op, **config_kwargs)
    state = QDMCControllerState.initial_state_for(config)
    return dynamics, config, state


def test_qdmc_converges_to_constant_setpoint_siso():
    dynamics, config, state = _siso_config(regularisation=0.1)
    closed_system = QDMCControllerClosedSystem(dynamics, config, state)

    r_traj = torch.tensor([[2.0]] * 60, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(
        r_traj, duration=30.0, dt=0.5, num_substeps=2
    )

    assert y_all[-1, 0].item() == pytest.approx(2.0, abs=0.1)


def test_qdmc_mimo_tracks_both_channels():
    dynamics, config, state = _mimo_config(regularisation=0.1)
    closed_system = QDMCControllerClosedSystem(dynamics, config, state)

    r_traj = torch.tensor([[1.5, -0.5]] * 60, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(
        r_traj, duration=30.0, dt=0.5, num_substeps=2
    )

    assert y_all[-1, 0].item() == pytest.approx(1.5, abs=0.15)
    assert y_all[-1, 1].item() == pytest.approx(-0.5, abs=0.15)


def test_qdmc_strict_policy_respects_u_bounds_throughout_simulation():
    dynamics, config, state = _siso_config(
        regularisation=0.1,
        u_min=torch.tensor([-1.0]),
        u_max=torch.tensor([1.0]),
    )
    closed_system = QDMCControllerClosedSystem(dynamics, config, state)

    r_traj = torch.tensor([[1000.0]] * 40, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(
        r_traj, duration=20.0, dt=0.5, num_substeps=2
    )

    assert (u_all[:, 0] <= 1.0 + 1e-3).all()
    assert (u_all[:, 0] >= -1.0 - 1e-3).all()


def test_qdmc_strict_policy_respects_y_bounds_for_feasible_setpoint():
    dynamics, config, state = _siso_config(
        regularisation=0.1,
        y_min=torch.tensor([-5.0]),
        y_max=torch.tensor([5.0]),
    )
    closed_system = QDMCControllerClosedSystem(dynamics, config, state)

    # Setpoint well within [y_min, y_max]: the strict-policy QP should
    # remain feasible throughout and never push y outside its bounds.
    r_traj = torch.tensor([[3.0]] * 40, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(
        r_traj, duration=20.0, dt=0.5, num_substeps=2
    )

    assert (y_all[:, 0] <= 5.0 + 1e-2).all()
    assert (y_all[:, 0] >= -5.0 - 1e-2).all()


def test_qdmc_respects_du_bounds_throughout_simulation():
    dynamics, config, state = _siso_config(
        regularisation=0.1,
        du_min=torch.tensor([-0.02]),
        du_max=torch.tensor([0.02]),
    )
    closed_system = QDMCControllerClosedSystem(dynamics, config, state)

    r_traj = torch.tensor([[1000.0]] * 40, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(
        r_traj, duration=20.0, dt=0.5, num_substeps=2
    )

    du = u_all[1:, 0] - u_all[:-1, 0]
    assert (du <= 0.02 + 1e-3).all()
    assert (du >= -0.02 - 1e-3).all()


def test_qdmc_soft_policy_runs_without_raising_when_target_outside_y_bounds():
    dynamics, config, state = _siso_config(
        regularisation=0.1,
        y_min=torch.tensor([-1.0]),
        y_max=torch.tensor([1.0]),
        policy="soft",
        rho_min=10.0,
        rho_max=10.0,
    )
    closed_system = QDMCControllerClosedSystem(dynamics, config, state)

    # Setpoint outside [y_min, y_max]; a strict policy might struggle, soft
    # should simply produce a (possibly bound-violating) solution.
    r_traj = torch.tensor([[10.0]] * 10, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(
        r_traj, duration=5.0, dt=0.5, num_substeps=1
    )

    assert t_all.shape[0] > 0


def test_qdmc_minimize_violation_policy_runs_without_raising():
    dynamics, config, state = _siso_config(
        regularisation=0.1,
        y_min=torch.tensor([-1.0]),
        y_max=torch.tensor([1.0]),
        policy="minimize_violation",
        rho_min=10.0,
        rho_max=10.0,
    )
    closed_system = QDMCControllerClosedSystem(dynamics, config, state)

    r_traj = torch.tensor([[10.0]] * 10, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(
        r_traj, duration=5.0, dt=0.5, num_substeps=1
    )

    assert t_all.shape[0] > 0


def test_qdmc_simulate_is_continuous_across_calls():
    dynamics, config, state = _siso_config(regularisation=0.1)
    closed_system = QDMCControllerClosedSystem(dynamics, config, state)

    r_traj1 = torch.tensor([[1.0]] * 40, dtype=torch.float64)
    t1, y1, u1 = closed_system.simulate(r_traj1, duration=10.0, dt=0.5, num_substeps=1)

    r_traj2 = torch.tensor([[1.0]] * 40, dtype=torch.float64)
    t2, y2, u2 = closed_system.simulate(r_traj2, duration=10.0, dt=0.5, num_substeps=1)

    assert t2[0].item() == pytest.approx(t1[-1].item())
    assert y2[0, 0].item() == pytest.approx(y1[-1, 0].item(), abs=1e-8)


def test_qdmc_warm_start_is_populated_after_simulation():
    dynamics, config, state = _siso_config(regularisation=0.1)
    closed_system = QDMCControllerClosedSystem(dynamics, config, state)

    r_traj = torch.tensor([[1.0]] * 20, dtype=torch.float64)
    closed_system.simulate(r_traj, duration=10.0, dt=0.5, num_substeps=1)

    assert closed_system.state.warm_start_x is not None


def test_qdmc_simulate_extends_short_r_traj_with_last_value():
    dynamics, config, state = _siso_config(regularisation=0.1)
    closed_system = QDMCControllerClosedSystem(dynamics, config, state)

    r_traj = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(
        r_traj, duration=10.0, dt=0.5, num_substeps=1
    )

    assert t_all.shape[0] > 0
    assert y_all.shape[0] == t_all.shape[0]
