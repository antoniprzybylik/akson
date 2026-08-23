import pytest
import torch

from akson import StateSpaceDynamics, OperatingPoint
from akson import (
    DMCRegulatorConfiguration,
    DMCRegulatorState,
    DMCRegulatorClosedSystem,
)


def _siso_step_response(D=5):
    # A simple monotonically saturating step response: 1 - exp(-0.5 i)
    vals = [1.0 - pow(2.71828182845904523536, -0.5 * i) for i in range(D)]
    return torch.tensor(vals, dtype=torch.float64).reshape(D, 1, 1)


def _stable_siso_dynamics():
    A = torch.tensor([[-1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    return StateSpaceDynamics.from_linear(A, B, C, D)


def _basic_config(N=3, Nu=2, D=5, **kwargs):
    S = _siso_step_response(D)
    op = OperatingPoint(
        u=torch.tensor([0.0], dtype=torch.float64),
        x=torch.tensor([0.0], dtype=torch.float64),
        y=torch.tensor([0.0], dtype=torch.float64),
    )
    return DMCRegulatorConfiguration(S, N, Nu, op, **kwargs)


def test_dmc_config_rejects_bad_step_response_ndim():
    S = torch.zeros(5, 1)  # should be 3D
    op = OperatingPoint(
        torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
    )
    with pytest.raises(ValueError):
        DMCRegulatorConfiguration(S, N=3, Nu=2, operating_point=op)


def test_dmc_config_rejects_empty_step_response():
    S = torch.zeros(0, 1, 1, dtype=torch.float64)
    op = OperatingPoint(
        torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="must not contain less than one sample"):
        DMCRegulatorConfiguration(S, N=3, Nu=2, operating_point=op)


def test_dmc_config_rejects_nonpositive_N():
    with pytest.raises(ValueError, match="Prediction horizon N"):
        _basic_config(N=0, Nu=2)


def test_dmc_config_rejects_nonpositive_Nu():
    with pytest.raises(ValueError, match="Control horizon Nu"):
        _basic_config(N=3, Nu=0)


def test_dmc_config_rejects_negative_regularisation():
    with pytest.raises(ValueError, match="Regularisation factor"):
        _basic_config(regularisation=-1.0)


def test_dmc_config_rejects_du_min_greater_than_du_max():
    with pytest.raises(ValueError):
        _basic_config(du_min=torch.tensor([1.0]), du_max=torch.tensor([-1.0]))


def test_dmc_config_rejects_u_min_greater_than_u_max():
    with pytest.raises(ValueError):
        _basic_config(u_min=torch.tensor([5.0]), u_max=torch.tensor([1.0]))


def test_dmc_config_stores_horizons_correctly():
    config = _basic_config(N=4, Nu=2, D=6)
    assert config.N == 4
    assert config.Nu == 2
    assert config.D == 6
    assert config.n_inputs == 1
    assert config.n_outputs == 1


def test_dmc_config_M_and_Mp_shapes():
    config = _basic_config(N=4, Nu=2, D=6)
    assert config.M.shape == (4, 2)
    assert config.Mp.shape == (4, 5)


def test_dmc_config_K_matches_regularised_least_squares_formula():
    config = _basic_config(N=4, Nu=2, D=6, regularisation=1.5)
    MTM = config.M.T @ config.M
    Lambda = 1.5 * torch.eye(config.Nu * config.n_inputs, dtype=config.dtype)
    expected_K = torch.linalg.solve(MTM + Lambda, config.M.T)
    assert torch.allclose(config.K, expected_K)


def test_dmc_config_sum_input_deltas_array_shape():
    config = _basic_config(N=4, Nu=2, D=6)
    assert config.sum_input_deltas_array.shape == (4, 2)


def test_dmc_state_zero_state_shapes():
    state = DMCRegulatorState.zero_state(dynamics_horizon=5, n_inputs=2)
    assert state.past_du.shape == (4, 2)
    assert state.current_u.shape == (2,)


def test_dmc_state_initial_state_for_matches_config():
    config = _basic_config(N=3, Nu=2, D=7)
    state = DMCRegulatorState.initial_state_for(config)
    assert state.past_du.shape == (6, 1)
    assert state.current_u.shape == (1,)


def test_dmc_closed_system_rejects_state_horizon_mismatch():
    config = _basic_config(N=3, Nu=2, D=7)
    bad_state = DMCRegulatorState.zero_state(dynamics_horizon=3, n_inputs=1)  # wrong horizon
    dynamics = _stable_siso_dynamics()
    with pytest.raises(ValueError, match="dynamics horizon"):
        DMCRegulatorClosedSystem(dynamics, config, bad_state)


def test_dmc_closed_system_rejects_state_input_count_mismatch():
    config = _basic_config(N=3, Nu=2, D=7)
    bad_state = DMCRegulatorState.zero_state(dynamics_horizon=7, n_inputs=3)  # wrong n_inputs
    dynamics = _stable_siso_dynamics()
    with pytest.raises(ValueError, match="Different assumed number of system inputs"):
        DMCRegulatorClosedSystem(dynamics, config, bad_state)


def test_dmc_closed_system_rejects_plant_input_mismatch():
    config = _basic_config(N=3, Nu=2, D=7)
    state = DMCRegulatorState.initial_state_for(config)

    A = torch.tensor([[-1.0, 0.0], [0.0, -2.0]], dtype=torch.float64)
    B = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    mimo_dynamics = StateSpaceDynamics.from_linear(A, B, C, D)

    with pytest.raises(ValueError, match="Different assumed number of system inputs"):
        DMCRegulatorClosedSystem(mimo_dynamics, config, state)


def test_dmc_closed_system_warns_when_regulator_u_min_looser_than_plant():
    S = _siso_step_response(5)
    op = OperatingPoint(
        torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
    )
    config = DMCRegulatorConfiguration(S, N=3, Nu=2, operating_point=op)  # no u_min at all
    state = DMCRegulatorState.initial_state_for(config)

    A = torch.tensor([[-1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    dynamics = StateSpaceDynamics.from_linear(A, B, C, D, u_min=torch.tensor([-1.0]), u_max=torch.tensor([1.0]))

    with pytest.warns(RuntimeWarning, match="u_min are looser"):
        DMCRegulatorClosedSystem(dynamics, config, state)


def test_dmc_closed_system_valid_construction_does_not_raise():
    config = _basic_config(N=3, Nu=2, D=7)
    state = DMCRegulatorState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = DMCRegulatorClosedSystem(dynamics, config, state)
    assert closed_system.config is config
    assert closed_system.state is state


def test_dmc_step_unconstrained_matches_K_times_error():
    config = _basic_config(N=3, Nu=2, D=5)  # no constraints at all
    state = DMCRegulatorState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = DMCRegulatorClosedSystem(dynamics, config, state)

    y = torch.tensor([0.0], dtype=torch.float64)
    r_traj = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float64)

    u_new = closed_system.step(y, r_traj)

    # current_u starts at 0 and operating_point.u is 0, so the free response
    # is just y repeated N times and du = K @ (r - y)
    e_flat = r_traj.reshape(-1) - y.repeat(config.N)
    expected_du = (config.K @ e_flat)[: config.n_inputs]
    assert torch.allclose(u_new, expected_du, atol=1e-10)


def test_dmc_step_clamps_du_to_du_max():
    config = _basic_config(N=3, Nu=2, D=5, du_min=torch.tensor([-0.05]), du_max=torch.tensor([0.05]))
    state = DMCRegulatorState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = DMCRegulatorClosedSystem(dynamics, config, state)

    y = torch.tensor([0.0], dtype=torch.float64)
    r_traj = torch.tensor([[100.0], [100.0], [100.0]], dtype=torch.float64)  # huge setpoint jump

    u_new = closed_system.step(y, r_traj)
    assert u_new.item() <= 0.05 + 1e-9


def test_dmc_step_clamps_u_to_u_max():
    config = _basic_config(
        N=3, Nu=2, D=5,
        u_min=torch.tensor([-1.0]), u_max=torch.tensor([1.0]),
    )
    state = DMCRegulatorState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = DMCRegulatorClosedSystem(dynamics, config, state)

    y = torch.tensor([0.0], dtype=torch.float64)
    r_traj = torch.tensor([[1000.0], [1000.0], [1000.0]], dtype=torch.float64)

    u_new = closed_system.step(y, r_traj)
    assert u_new.item() <= 1.0 + 1e-9


def test_dmc_step_updates_past_du_history():
    config = _basic_config(N=3, Nu=2, D=5)
    state = DMCRegulatorState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = DMCRegulatorClosedSystem(dynamics, config, state)

    y = torch.tensor([0.0], dtype=torch.float64)
    r_traj = torch.tensor([[0.5], [0.5], [0.5]], dtype=torch.float64)
    closed_system.step(y, r_traj)

    assert closed_system.state.past_du.shape == (4, 1)


def test_dmc_step_rejects_wrong_y_shape():
    config = _basic_config(N=3, Nu=2, D=5)
    state = DMCRegulatorState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = DMCRegulatorClosedSystem(dynamics, config, state)

    bad_y = torch.tensor([0.0, 0.0], dtype=torch.float64)
    r_traj = torch.tensor([[0.5], [0.5], [0.5]], dtype=torch.float64)
    with pytest.raises(ValueError):
        closed_system.step(bad_y, r_traj)


def test_dmc_step_rejects_wrong_r_traj_shape():
    config = _basic_config(N=3, Nu=2, D=5)
    state = DMCRegulatorState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = DMCRegulatorClosedSystem(dynamics, config, state)

    y = torch.tensor([0.0], dtype=torch.float64)
    bad_r_traj = torch.tensor([[0.5], [0.5]], dtype=torch.float64)  # should have length N=3
    with pytest.raises(ValueError):
        closed_system.step(y, bad_r_traj)


def test_dmc_step_with_polishing_respects_constraints():
    config = _basic_config(
        N=3, Nu=2, D=5,
        du_min=torch.tensor([-0.05]), du_max=torch.tensor([0.05]),
        u_min=torch.tensor([-1.0]), u_max=torch.tensor([1.0]),
        use_polishing=True,
    )
    state = DMCRegulatorState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = DMCRegulatorClosedSystem(dynamics, config, state)

    y = torch.tensor([0.0], dtype=torch.float64)
    r_traj = torch.tensor([[100.0], [100.0], [100.0]], dtype=torch.float64)
    u_new = closed_system.step(y, r_traj)

    assert u_new.item() <= 1.0 + 1e-6


def test_dmc_simulate_rejects_num_substeps_below_one():
    config = _basic_config(N=3, Nu=2, D=5)
    state = DMCRegulatorState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = DMCRegulatorClosedSystem(dynamics, config, state)

    r_traj = torch.tensor([[1.0]] * 10, dtype=torch.float64)
    with pytest.raises(ValueError, match="num_substeps must be at least 1"):
        closed_system.simulate(r_traj, duration=5.0, dt=1.0, num_substeps=0)


def test_dmc_simulate_rejects_wrong_r_traj_shape():
    config = _basic_config(N=3, Nu=2, D=5)
    state = DMCRegulatorState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = DMCRegulatorClosedSystem(dynamics, config, state)

    bad_r_traj = torch.tensor([[1.0, 2.0]] * 10, dtype=torch.float64)  # wrong n_outputs
    with pytest.raises(ValueError):
        closed_system.simulate(bad_r_traj, duration=5.0, dt=1.0, num_substeps=1)


def test_dmc_simulate_returns_consistent_shapes():
    config = _basic_config(N=3, Nu=2, D=5)
    state = DMCRegulatorState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = DMCRegulatorClosedSystem(dynamics, config, state)

    r_traj = torch.tensor([[1.0]] * 10, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(r_traj, duration=5.0, dt=1.0, num_substeps=2)

    assert t_all.shape[0] == y_all.shape[0]
    assert y_all.shape[1] == 1
    assert u_all.shape[1] == 1
    assert u_all.shape[0] == 6  # 5 control steps + initial value
