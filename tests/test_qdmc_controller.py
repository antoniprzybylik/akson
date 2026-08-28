import pytest
import torch

from akson import StateSpaceDynamics, OperatingPoint
from akson import (
    QDMCControllerConfiguration,
    QDMCControllerState,
    QDMCControllerClosedSystem,
    DMCControllerConfiguration,
    DMCControllerState,
    DMCControllerClosedSystem,
)


def _siso_step_response(D=5):
    vals = [1.0 - pow(2.71828182845904523536, -0.5 * i) for i in range(D)]
    return torch.tensor(vals, dtype=torch.float64).reshape(D, 1, 1)


def _stable_siso_dynamics(**kwargs):
    A = torch.tensor([[-1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    return StateSpaceDynamics.from_linear(A, B, C, D, **kwargs)


def _zero_op():
    return OperatingPoint(
        u=torch.tensor([0.0], dtype=torch.float64),
        x=torch.tensor([0.0], dtype=torch.float64),
        y=torch.tensor([0.0], dtype=torch.float64),
    )


def _basic_config(N=3, Nu=2, D=5, **kwargs):
    S = _siso_step_response(D)
    return QDMCControllerConfiguration(S, N, Nu, _zero_op(), **kwargs)


def test_qdmc_config_rejects_nonpositive_N():
    with pytest.raises(ValueError, match="Prediction horizon N"):
        _basic_config(N=0)


def test_qdmc_config_rejects_nonpositive_Nu():
    with pytest.raises(ValueError, match="Control horizon Nu"):
        _basic_config(Nu=0)


def test_qdmc_config_rejects_negative_regularisation():
    with pytest.raises(ValueError, match="Regularisation factor"):
        _basic_config(regularisation=-1.0)


def test_qdmc_config_rejects_negative_rho_min():
    with pytest.raises(ValueError, match="rho_min must be nonnegative"):
        _basic_config(policy="soft", rho_min=-1.0, rho_max=1.0)


def test_qdmc_config_rejects_negative_rho_max():
    with pytest.raises(ValueError, match="rho_max must be nonnegative"):
        _basic_config(policy="soft", rho_min=1.0, rho_max=-1.0)


def test_qdmc_config_rejects_invalid_policy():
    with pytest.raises(ValueError, match="Invalid policy"):
        _basic_config(policy="bogus")


def test_qdmc_config_rejects_rho_with_strict_policy():
    with pytest.raises(ValueError, match="are invalid for policy"):
        _basic_config(policy="strict", rho_min=1.0, rho_max=1.0)


def test_qdmc_config_rejects_soft_policy_without_rho():
    with pytest.raises(ValueError, match="are required for policy"):
        _basic_config(policy="soft")


def test_qdmc_config_rejects_minimize_violation_policy_without_rho():
    with pytest.raises(ValueError, match="are required for policy"):
        _basic_config(policy="minimize_violation")


def test_qdmc_config_accepts_soft_policy_with_rho():
    config = _basic_config(policy="soft", rho_min=1.0, rho_max=2.0)
    assert config.policy == "soft"
    assert config.rho_min == 1.0
    assert config.rho_max == 2.0


def test_qdmc_config_rejects_y_min_greater_than_y_max():
    with pytest.raises(ValueError):
        _basic_config(y_min=torch.tensor([5.0]), y_max=torch.tensor([1.0]))


def test_qdmc_config_default_policy_is_strict():
    config = _basic_config()
    assert config.policy == "strict"


def test_qdmc_config_M_and_Mp_shapes():
    config = _basic_config(N=4, Nu=2, D=6)
    assert config.M.shape == (4, 2)
    assert config.Mp.shape == (4, 5)


def test_qdmc_state_initial_state_for_matches_config():
    config = _basic_config(N=3, Nu=2, D=7)
    state = QDMCControllerState.initial_state_for(config)
    assert state.past_du.shape == (6, 1)
    assert state.current_u.shape == (1,)
    assert state.warm_start_x is None


def test_qdmc_state_rejects_wrong_warm_start_shape():
    config = _basic_config(N=3, Nu=2, D=5)
    past_du = torch.zeros(4, 1, dtype=torch.float64)
    current_u = torch.zeros(1, dtype=torch.float64)
    bad_warm_start = torch.zeros(99, dtype=torch.float64)
    with pytest.raises(ValueError):
        QDMCControllerState(past_du, current_u, config, warm_start_x=bad_warm_start)


def test_qdmc_closed_system_rejects_state_horizon_mismatch():
    config = _basic_config(N=3, Nu=2, D=7)
    bad_past_du = torch.zeros(2, 1, dtype=torch.float64)  # wrong horizon (should be 6)
    bad_state = QDMCControllerState(
        bad_past_du, torch.zeros(1, dtype=torch.float64), config
    )
    dynamics = _stable_siso_dynamics()
    with pytest.raises(ValueError, match="dynamics horizon"):
        QDMCControllerClosedSystem(dynamics, config, bad_state)


def test_qdmc_closed_system_rejects_plant_input_mismatch():
    config = _basic_config(N=3, Nu=2, D=7)
    state = QDMCControllerState.initial_state_for(config)

    A = torch.tensor([[-1.0, 0.0], [0.0, -2.0]], dtype=torch.float64)
    B = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    mimo_dynamics = StateSpaceDynamics.from_linear(A, B, C, D)

    with pytest.raises(ValueError, match="Different assumed number of system inputs"):
        QDMCControllerClosedSystem(mimo_dynamics, config, state)


def test_qdmc_closed_system_warns_when_controller_u_max_looser_than_plant():
    config = _basic_config()  # no u_max at all
    state = QDMCControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics(
        u_min=torch.tensor([-1.0]), u_max=torch.tensor([1.0])
    )

    with pytest.warns(RuntimeWarning, match="looser"):
        QDMCControllerClosedSystem(dynamics, config, state)


def test_qdmc_closed_system_valid_construction_does_not_raise():
    config = _basic_config(N=3, Nu=2, D=7)
    state = QDMCControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = QDMCControllerClosedSystem(dynamics, config, state)
    assert closed_system.config is config
    assert closed_system.state is state


def test_qdmc_step_rejects_wrong_y_shape():
    config = _basic_config(N=3, Nu=2, D=5)
    state = QDMCControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = QDMCControllerClosedSystem(dynamics, config, state)

    bad_y = torch.tensor([0.0, 0.0], dtype=torch.float64)
    r_traj = torch.tensor([[0.5], [0.5], [0.5]], dtype=torch.float64)
    with pytest.raises(ValueError):
        closed_system.step(bad_y, r_traj)


def test_qdmc_step_rejects_wrong_r_traj_shape():
    config = _basic_config(N=3, Nu=2, D=5)
    state = QDMCControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = QDMCControllerClosedSystem(dynamics, config, state)

    y = torch.tensor([0.0], dtype=torch.float64)
    bad_r_traj = torch.tensor(
        [[0.5], [0.5]], dtype=torch.float64
    )  # should have length N=3
    with pytest.raises(ValueError):
        closed_system.step(y, bad_r_traj)


def test_qdmc_step_unconstrained_matches_dmc_solution():
    # With no u/du/y constraints at all, the strict-policy QP has the same
    # unconstrained normal-equations solution as classical DMC.
    N, Nu, D = 3, 2, 5
    S = _siso_step_response(D)

    dmc_config = DMCControllerConfiguration(S, N, Nu, _zero_op())
    dmc_state = DMCControllerState.initial_state_for(dmc_config)
    dmc_closed = DMCControllerClosedSystem(
        _stable_siso_dynamics(), dmc_config, dmc_state
    )

    qdmc_config = QDMCControllerConfiguration(S, N, Nu, _zero_op())
    qdmc_state = QDMCControllerState.initial_state_for(qdmc_config)
    qdmc_closed = QDMCControllerClosedSystem(
        _stable_siso_dynamics(), qdmc_config, qdmc_state
    )

    y = torch.tensor([0.0], dtype=torch.float64)
    r_traj = torch.tensor([[0.5], [0.5], [0.5]], dtype=torch.float64)

    u_dmc = dmc_closed.step(y, r_traj)
    u_qdmc = qdmc_closed.step(y, r_traj)

    assert u_dmc.item() == pytest.approx(u_qdmc.item(), abs=5e-3)


def test_qdmc_step_respects_u_max_strict_policy():
    config = _basic_config(
        N=3,
        Nu=2,
        D=5,
        u_min=torch.tensor([-1.0]),
        u_max=torch.tensor([1.0]),
    )
    state = QDMCControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = QDMCControllerClosedSystem(dynamics, config, state)

    y = torch.tensor([0.0], dtype=torch.float64)
    r_traj = torch.tensor([[1000.0], [1000.0], [1000.0]], dtype=torch.float64)

    u_new = closed_system.step(y, r_traj)
    assert u_new.item() <= 1.0 + 1e-3


def test_qdmc_step_updates_past_du_and_warm_start():
    config = _basic_config(N=3, Nu=2, D=5)
    state = QDMCControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = QDMCControllerClosedSystem(dynamics, config, state)

    y = torch.tensor([0.0], dtype=torch.float64)
    r_traj = torch.tensor([[0.5], [0.5], [0.5]], dtype=torch.float64)
    closed_system.step(y, r_traj)

    assert closed_system.state.warm_start_x is not None
    assert closed_system.state.past_du.shape == (4, 1)


def test_qdmc_simulate_rejects_num_substeps_below_one():
    config = _basic_config(N=3, Nu=2, D=5)
    state = QDMCControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = QDMCControllerClosedSystem(dynamics, config, state)

    r_traj = torch.tensor([[1.0]] * 10, dtype=torch.float64)
    with pytest.raises(ValueError, match="num_substeps must be at least 1"):
        closed_system.simulate(r_traj, duration=5.0, dt=1.0, num_substeps=0)


def test_qdmc_simulate_rejects_wrong_r_traj_shape():
    config = _basic_config(N=3, Nu=2, D=5)
    state = QDMCControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = QDMCControllerClosedSystem(dynamics, config, state)

    bad_r_traj = torch.tensor([[1.0, 2.0]] * 10, dtype=torch.float64)
    with pytest.raises(ValueError):
        closed_system.simulate(bad_r_traj, duration=5.0, dt=1.0, num_substeps=1)


def test_qdmc_simulate_returns_consistent_shapes():
    config = _basic_config(N=3, Nu=2, D=5)
    state = QDMCControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = QDMCControllerClosedSystem(dynamics, config, state)

    r_traj = torch.tensor([[0.5]] * 6, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(
        r_traj, duration=3.0, dt=1.0, num_substeps=2
    )

    assert t_all.shape[0] == y_all.shape[0]
    assert y_all.shape[1] == 1
    assert u_all.shape[1] == 1
    assert u_all.shape[0] == 4  # 3 control steps + initial value
