import pytest
import torch

from akson import StateSpaceDynamics, StateSpaceSystem


def fresh_system():
    """
    dx/dt = -x + u
    y = x

    Stable first-order system with known behavior.
    """
    A = torch.tensor([[-1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)

    dynamics = StateSpaceDynamics.from_linear(A, B, C, D)
    return StateSpaceSystem(dynamics)


@pytest.fixture
def first_order_system():
    return fresh_system()


def constant_input(value):
    def u(_t):
        return torch.tensor([value], dtype=torch.float64)

    return u


def test_initial_state_and_time():
    system = fresh_system()

    assert torch.allclose(system.x, torch.tensor([0.0], dtype=torch.float64))
    assert system.simulation_time == 0.0


def test_state_and_time_after_reset(first_order_system):
    system = first_order_system
    system.reset()

    assert torch.allclose(system.x, torch.tensor([0.0], dtype=torch.float64))
    assert system.simulation_time == 0.0


def test_reset_restores_initial_state_and_time(first_order_system):
    system = first_order_system
    system.reset()

    system.simulate(
        constant_input(1.0),
        duration=1.0,
        step_size=0.1,
    )

    assert system.simulation_time > 0
    assert not torch.allclose(system.x, torch.tensor([0.0], dtype=torch.float64))

    system.reset()

    assert system.simulation_time == 0.0
    assert torch.allclose(
        system.x,
        torch.tensor([0.0], dtype=torch.float64),
    )


def test_reset_with_custom_state(first_order_system):
    system = first_order_system
    new_state = torch.tensor([5.0], dtype=torch.float64)
    system.reset(new_state)

    assert torch.allclose(system.x, new_state)
    assert system.simulation_time == 0.0


def test_duration_validation_1(first_order_system):
    system = first_order_system

    with pytest.raises(ValueError):
        system.simulate(
            constant_input(1.0),
            duration=0.0,
        )


def test_duration_validation_2(first_order_system):
    system = first_order_system

    with pytest.raises(ValueError):
        system.simulate(
            constant_input(1.0),
            duration=-1.0,
        )


def test_custom_input_shape_is_checked(first_order_system):
    system = first_order_system

    def bad_input(_t):
        return torch.tensor([1.0, 2.0], dtype=torch.float64)

    with pytest.raises(ValueError):
        system.simulate(
            bad_input,
            duration=1.0,
            step_size=0.1,
        )


def test_state_is_independent_copy(first_order_system):
    system = first_order_system

    _, x, _ = system.simulate(
        constant_input(1.0),
        duration=1.0,
        step_size=0.1,
    )

    old_state = system.x.clone()

    x[-1] += 100

    assert torch.allclose(
        system.x,
        old_state,
    )


def test_reset_rejects_wrong_shape(first_order_system):
    system = first_order_system

    with pytest.raises(ValueError):
        system.reset(torch.tensor([1.0, 2.0]))
