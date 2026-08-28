import pytest
import torch

from akson import StateSpaceDynamics
from akson import (
    PIDChannel,
    PIDControllerConfiguration,
    PIDControllerState,
    PIDControllerClosedSystem,
)


def _stable_siso_dynamics():
    # dx/dt = -x + u, y = x
    A = torch.tensor([[-1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    return StateSpaceDynamics.from_linear(A, B, C, D)


def test_pid_channel_valid_construction():
    channel = PIDChannel(output_idx=0, input_idx=0, K=1.0, Ti=5.0, Td=0.0)
    assert channel.output_idx == 0
    assert channel.input_idx == 0


def test_pid_channel_rejects_negative_output_idx():
    with pytest.raises(ValueError, match="output_idx"):
        PIDChannel(output_idx=-1, input_idx=0, K=1.0, Ti=5.0, Td=0.0)


def test_pid_channel_rejects_negative_input_idx():
    with pytest.raises(ValueError, match="input_idx"):
        PIDChannel(output_idx=0, input_idx=-1, K=1.0, Ti=5.0, Td=0.0)


def test_pid_channel_rejects_nonpositive_Ti():
    with pytest.raises(ValueError, match="Ti must be positive"):
        PIDChannel(output_idx=0, input_idx=0, K=1.0, Ti=0.0, Td=0.0)


def test_pid_channel_rejects_negative_Td():
    with pytest.raises(ValueError, match="Td must be nonnegative"):
        PIDChannel(output_idx=0, input_idx=0, K=1.0, Ti=5.0, Td=-1.0)


def test_pid_channel_allows_zero_Td():
    channel = PIDChannel(output_idx=0, input_idx=0, K=1.0, Ti=5.0, Td=0.0)
    assert channel.Td == 0.0


def test_pid_config_rejects_channel_with_nonexisting_input():
    channel = PIDChannel(output_idx=0, input_idx=5, K=1.0, Ti=1.0, Td=0.0)
    with pytest.raises(ValueError, match="nonexisting input"):
        PIDControllerConfiguration(n_inputs=1, n_outputs=1, channels=[channel], t=0.1)


def test_pid_config_rejects_channel_with_nonexisting_output():
    channel = PIDChannel(output_idx=5, input_idx=0, K=1.0, Ti=1.0, Td=0.0)
    with pytest.raises(ValueError, match="nonexisting output"):
        PIDControllerConfiguration(n_inputs=1, n_outputs=1, channels=[channel], t=0.1)


def test_pid_config_rejects_duplicate_channel():
    channel1 = PIDChannel(output_idx=0, input_idx=0, K=1.0, Ti=1.0, Td=0.0)
    channel2 = PIDChannel(output_idx=0, input_idx=0, K=2.0, Ti=2.0, Td=0.0)
    with pytest.raises(ValueError, match="defined twice"):
        PIDControllerConfiguration(n_inputs=1, n_outputs=1, channels=[channel1, channel2], t=0.1)


def test_pid_config_accepts_multiple_distinct_channels():
    channel1 = PIDChannel(output_idx=0, input_idx=0, K=1.0, Ti=1.0, Td=0.0)
    channel2 = PIDChannel(output_idx=1, input_idx=1, K=-1.0, Ti=2.0, Td=0.0)
    config = PIDControllerConfiguration(n_inputs=2, n_outputs=2, channels=[channel1, channel2], t=0.1)
    assert config.n_inputs == 2
    assert config.n_outputs == 2


def test_pid_config_rejects_u_min_greater_than_u_max():
    channel = PIDChannel(output_idx=0, input_idx=0, K=1.0, Ti=1.0, Td=0.0)
    with pytest.raises(ValueError):
        PIDControllerConfiguration(
            n_inputs=1, n_outputs=1, channels=[channel], t=0.1,
            u_min=torch.tensor([5.0]), u_max=torch.tensor([1.0]),
        )


def test_pid_config_rejects_u0_outside_bounds():
    channel = PIDChannel(output_idx=0, input_idx=0, K=1.0, Ti=1.0, Td=0.0)
    with pytest.raises(ValueError):
        PIDControllerConfiguration(
            n_inputs=1, n_outputs=1, channels=[channel], t=0.1,
            u0=torch.tensor([10.0]),
            u_min=torch.tensor([0.0]), u_max=torch.tensor([5.0]),
        )


def test_pid_config_default_u0_is_none():
    channel = PIDChannel(output_idx=0, input_idx=0, K=1.0, Ti=1.0, Td=0.0)
    config = PIDControllerConfiguration(n_inputs=1, n_outputs=1, channels=[channel], t=0.1)
    assert config.u0 is None


def test_pid_config_coefficients_formula():
    K, Ti, Td, t = 2.0, 5.0, 0.1, 0.5
    channel = PIDChannel(output_idx=0, input_idx=0, K=K, Ti=Ti, Td=Td)
    config = PIDControllerConfiguration(n_inputs=1, n_outputs=1, channels=[channel], t=t)

    expected_r0 = K * (1.0 + t / (2.0 * Ti) + Td / t)
    expected_r1 = K * (t / (2.0 * Ti) - 2.0 * Td / t - 1.0)
    expected_r2 = K * Td / t

    assert config.coeffs[0, 0, 0].item() == pytest.approx(expected_r0)
    assert config.coeffs[0, 0, 1].item() == pytest.approx(expected_r1)
    assert config.coeffs[0, 0, 2].item() == pytest.approx(expected_r2)


def test_pid_config_coefficients_are_zero_for_unused_channel_pairs():
    channel = PIDChannel(output_idx=0, input_idx=0, K=1.0, Ti=1.0, Td=0.0)
    config = PIDControllerConfiguration(n_inputs=2, n_outputs=2, channels=[channel], t=0.1)
    # input_idx=1 / output_idx=1 was never configured, so its coefficients stay zero
    assert torch.allclose(config.coeffs[1, 1], torch.zeros(3, dtype=config.dtype))
    assert torch.allclose(config.coeffs[0, 1], torch.zeros(3, dtype=config.dtype))


def test_pid_state_defaults_are_zero():
    state = PIDControllerState(n_inputs=2, n_outputs=3)
    assert torch.allclose(state.e_prev, torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(state.e_prev_prev, torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(state.u_prev, torch.zeros(2, dtype=torch.float64))


def test_pid_state_initial_state_for_matches_configuration():
    channel = PIDChannel(output_idx=0, input_idx=0, K=1.0, Ti=1.0, Td=0.0)
    config = PIDControllerConfiguration(n_inputs=2, n_outputs=3, channels=[channel], t=0.1)
    state = PIDControllerState.initial_state_for(config)
    assert state.e_prev.shape == (3,)
    assert state.u_prev.shape == (2,)


def test_pid_step_unconstrained_matches_formula():
    K, Ti, Td, t = 1.0, 10.0, 0.0, 0.5
    channel = PIDChannel(output_idx=0, input_idx=0, K=K, Ti=Ti, Td=Td)
    config = PIDControllerConfiguration(
        n_inputs=1, n_outputs=1, channels=[channel], t=t, u0=torch.tensor([0.0])
    )
    state = PIDControllerState.initial_state_for(config)

    dynamics = _stable_siso_dynamics()
    closed_system = PIDControllerClosedSystem(
        dynamics, torch.tensor([0.0], dtype=torch.float64), config, state
    )

    y = torch.tensor([0.0], dtype=torch.float64)
    setpoint = torch.tensor([1.0], dtype=torch.float64)
    u_new = closed_system.step(y, setpoint)

    r0 = K * (1.0 + t / (2.0 * Ti) + Td / t)
    assert u_new.item() == pytest.approx(r0)


def test_pid_step_updates_error_history():
    channel = PIDChannel(output_idx=0, input_idx=0, K=1.0, Ti=10.0, Td=0.0)
    config = PIDControllerConfiguration(
        n_inputs=1, n_outputs=1, channels=[channel], t=0.5, u0=torch.tensor([0.0])
    )
    state = PIDControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = PIDControllerClosedSystem(
        dynamics, torch.tensor([0.0], dtype=torch.float64), config, state
    )

    y = torch.tensor([0.0], dtype=torch.float64)
    setpoint = torch.tensor([1.0], dtype=torch.float64)
    closed_system.step(y, setpoint)

    assert closed_system.state.e_prev.item() == pytest.approx(1.0)
    assert closed_system.state.e_prev_prev.item() == pytest.approx(0.0)


def test_pid_step_clamps_to_u_max_relative_to_u0():
    channel = PIDChannel(output_idx=0, input_idx=0, K=100.0, Ti=1.0, Td=0.0)
    config = PIDControllerConfiguration(
        n_inputs=1, n_outputs=1, channels=[channel], t=1.0,
        u0=torch.tensor([0.0]),
        u_min=torch.tensor([-1.0]), u_max=torch.tensor([1.0]),
    )
    state = PIDControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = PIDControllerClosedSystem(
        dynamics, torch.tensor([0.0], dtype=torch.float64), config, state
    )

    y = torch.tensor([0.0], dtype=torch.float64)
    setpoint = torch.tensor([1000.0], dtype=torch.float64)  # huge error forces clamp
    u_new = closed_system.step(y, setpoint)

    assert u_new.item() == pytest.approx(1.0)


def test_pid_step_clamps_to_u_min_relative_to_u0():
    channel = PIDChannel(output_idx=0, input_idx=0, K=100.0, Ti=1.0, Td=0.0)
    config = PIDControllerConfiguration(
        n_inputs=1, n_outputs=1, channels=[channel], t=1.0,
        u0=torch.tensor([0.0]),
        u_min=torch.tensor([-1.0]), u_max=torch.tensor([1.0]),
    )
    state = PIDControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = PIDControllerClosedSystem(
        dynamics, torch.tensor([0.0], dtype=torch.float64), config, state
    )

    y = torch.tensor([0.0], dtype=torch.float64)
    setpoint = torch.tensor([-1000.0], dtype=torch.float64)
    u_new = closed_system.step(y, setpoint)

    assert u_new.item() == pytest.approx(-1.0)


def test_pid_step_rejects_wrong_y_shape():
    channel = PIDChannel(output_idx=0, input_idx=0, K=1.0, Ti=1.0, Td=0.0)
    config = PIDControllerConfiguration(n_inputs=1, n_outputs=1, channels=[channel], t=0.1)
    state = PIDControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = PIDControllerClosedSystem(
        dynamics, torch.tensor([0.0], dtype=torch.float64), config, state
    )

    bad_y = torch.tensor([0.0, 0.0], dtype=torch.float64)
    setpoint = torch.tensor([1.0], dtype=torch.float64)
    with pytest.raises(ValueError):
        closed_system.step(bad_y, setpoint)


def test_pid_step_rejects_wrong_setpoint_shape():
    channel = PIDChannel(output_idx=0, input_idx=0, K=1.0, Ti=1.0, Td=0.0)
    config = PIDControllerConfiguration(n_inputs=1, n_outputs=1, channels=[channel], t=0.1)
    state = PIDControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = PIDControllerClosedSystem(
        dynamics, torch.tensor([0.0], dtype=torch.float64), config, state
    )

    y = torch.tensor([0.0], dtype=torch.float64)
    bad_setpoint = torch.tensor([1.0, 2.0], dtype=torch.float64)
    with pytest.raises(ValueError):
        closed_system.step(y, bad_setpoint)


def test_pid_closed_system_rejects_mismatched_n_inputs():
    channel = PIDChannel(output_idx=0, input_idx=0, K=1.0, Ti=1.0, Td=0.0)
    config = PIDControllerConfiguration(n_inputs=2, n_outputs=1, channels=[channel], t=0.1)
    state = PIDControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()  # n_inputs=1
    with pytest.raises(ValueError, match="Different assumed number of system inputs"):
        PIDControllerClosedSystem(
            dynamics, torch.tensor([0.0], dtype=torch.float64), config, state
        )


def test_pid_closed_system_rejects_mismatched_n_outputs():
    channel = PIDChannel(output_idx=0, input_idx=0, K=1.0, Ti=1.0, Td=0.0)
    config = PIDControllerConfiguration(n_inputs=1, n_outputs=2, channels=[channel], t=0.1)
    state = PIDControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()  # n_outputs=1
    with pytest.raises(ValueError, match="Different assumed number of system outputs"):
        PIDControllerClosedSystem(
            dynamics, torch.tensor([0.0], dtype=torch.float64), config, state
        )


def test_pid_simulate_rejects_num_substeps_below_one():
    channel = PIDChannel(output_idx=0, input_idx=0, K=1.0, Ti=1.0, Td=0.0)
    config = PIDControllerConfiguration(n_inputs=1, n_outputs=1, channels=[channel], t=0.1)
    state = PIDControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = PIDControllerClosedSystem(
        dynamics, torch.tensor([0.0], dtype=torch.float64), config, state
    )
    setpoints = torch.tensor([[1.0]] * 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="num_substeps must be at least 1"):
        closed_system.simulate(setpoints, duration=1.0, num_substeps=0)


def test_pid_simulate_rejects_wrong_setpoints_shape():
    channel = PIDChannel(output_idx=0, input_idx=0, K=1.0, Ti=1.0, Td=0.0)
    config = PIDControllerConfiguration(n_inputs=1, n_outputs=1, channels=[channel], t=0.1)
    state = PIDControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = PIDControllerClosedSystem(
        dynamics, torch.tensor([0.0], dtype=torch.float64), config, state
    )
    bad_setpoints = torch.tensor([[1.0, 2.0]] * 5, dtype=torch.float64)  # wrong n_outputs
    with pytest.raises(ValueError):
        closed_system.simulate(bad_setpoints, duration=1.0, num_substeps=2)


def test_pid_simulate_returns_consistent_shapes():
    channel = PIDChannel(output_idx=0, input_idx=0, K=0.5, Ti=5.0, Td=0.0)
    t = 0.1
    config = PIDControllerConfiguration(
        n_inputs=1, n_outputs=1, channels=[channel], t=t, u0=torch.tensor([0.0])
    )
    state = PIDControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = PIDControllerClosedSystem(
        dynamics, torch.tensor([0.0], dtype=torch.float64), config, state
    )
    setpoints = torch.tensor([[1.0]] * 10, dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(setpoints, duration=1.0, num_substeps=2)

    assert t_all.shape[0] == y_all.shape[0]
    assert y_all.shape[1] == 1
    assert u_all.shape[1] == 1
    # 10 controller steps -> 11 control inputs recorded (including the initial one)
    assert u_all.shape[0] == 11


def test_pid_simulate_pads_short_setpoints_with_last_value():
    channel = PIDChannel(output_idx=0, input_idx=0, K=0.5, Ti=5.0, Td=0.0)
    t = 0.1
    config = PIDControllerConfiguration(
        n_inputs=1, n_outputs=1, channels=[channel], t=t, u0=torch.tensor([0.0])
    )
    state = PIDControllerState.initial_state_for(config)
    dynamics = _stable_siso_dynamics()
    closed_system = PIDControllerClosedSystem(
        dynamics, torch.tensor([0.0], dtype=torch.float64), config, state
    )
    # Only 2 setpoints given, but 10 steps needed -> should be padded, not raise
    setpoints = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    t_all, y_all, u_all = closed_system.simulate(setpoints, duration=1.0, num_substeps=1)
    assert t_all.shape[0] > 0
