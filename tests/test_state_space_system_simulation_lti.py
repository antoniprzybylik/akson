import pytest
import torch

from akson import StateSpaceSystem, StateSpaceDynamics


@pytest.fixture
def first_order_system():
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


def constant_input(value):
    def u(_t):
        return torch.tensor([value], dtype=torch.float64)

    return u


def test_first_simulation_returns_absolute_time_starting_at_zero(first_order_system):
    system = first_order_system
    system.reset()

    t, x, y = system.simulate(
        constant_input(1.0),
        duration=1.0,
        step_size=0.1,
    )

    assert t[0].item() == pytest.approx(0.0)

    assert x.shape[0] == t.shape[0]
    assert y.shape[0] == t.shape[0]

    assert system.simulation_time == pytest.approx(1.0)


def test_state_is_updated_after_simulation(first_order_system):
    system = first_order_system
    system.reset()

    _, x, _ = system.simulate(
        constant_input(1.0),
        duration=1.0,
        step_size=0.1,
    )

    assert torch.allclose(
        system.x,
        x[-1],
    )


def test_first_trajectory_point_is_current_state(first_order_system):
    system = first_order_system

    initial = torch.tensor([2.0], dtype=torch.float64)
    system.reset(initial)

    _, x, _ = system.simulate(
        constant_input(0.0),
        duration=1.0,
        step_size=0.1,
    )

    assert torch.allclose(
        x[0],
        initial,
    )


def test_second_simulation_continues_state(first_order_system):
    system = first_order_system
    system.reset()

    t1, x1, _ = system.simulate(
        constant_input(1.0),
        duration=1.0,
        step_size=0.1,
    )

    state_after_first = system.x.clone()

    t2, x2, _ = system.simulate(
        constant_input(1.0),
        duration=1.0,
        step_size=0.1,
    )

    # Time should continue
    assert t2[0].item() == pytest.approx(1.0)
    assert t2[-1].item() == pytest.approx(2.0)

    # First point of second trajectory must be previous final state
    assert torch.allclose(
        x2[0],
        state_after_first,
        atol=1e-10,
    )

    # Final time should be cumulative
    assert system.simulation_time == pytest.approx(2.0)


def test_simulation_passes_absolute_time_to_input_function(first_order_system):
    system = first_order_system
    system.reset()

    seen_times = []

    def u(t):
        seen_times.append(t)
        return torch.tensor([1.0], dtype=torch.float64)

    system.simulate(
        u,
        duration=1.0,
        step_size=0.25,
    )

    assert seen_times[0] == pytest.approx(0.0)
    assert seen_times[-1] == pytest.approx(1.0)


def test_continuation_matches_single_long_simulation(first_order_system):
    """
    Two chained simulations should produce the same final state
    as one longer simulation.
    """
    system1 = first_order_system
    system1.reset()

    system1.simulate(
        constant_input(1.0),
        duration=1.0,
        step_size=0.05,
    )

    system1.simulate(
        constant_input(1.0),
        duration=1.0,
        step_size=0.05,
    )

    final_split = system1.x.clone()

    system2 = first_order_system
    system2.reset()

    _, x_long, _ = system2.simulate(
        constant_input(1.0),
        duration=2.0,
        step_size=0.05,
    )

    final_long = x_long[-1]

    assert torch.allclose(
        final_split,
        final_long,
        atol=1e-8,
    )


def test_reset_allows_reproducible_simulation(first_order_system):
    system = first_order_system

    _, x1, _ = system.simulate(
        constant_input(1.0),
        duration=1.0,
        step_size=0.05,
    )

    system.reset()

    _, x2, _ = system.simulate(
        constant_input(1.0),
        duration=1.0,
        step_size=0.05,
    )

    assert torch.allclose(
        x1,
        x2,
        atol=1e-12,
    )


def test_second_simulation_uses_existing_state(first_order_system):
    system = first_order_system

    system.simulate(
        constant_input(1.0),
        duration=1.0,
        step_size=0.05,
    )

    state_after_first = system.x.clone()

    system.simulate(
        constant_input(0.0),
        duration=0.5,
        step_size=0.05,
    )

    # Decays from previous state, not from zero
    assert system.x < state_after_first
    assert system.x > 0


def test_simulation_time_is_monotonic(first_order_system):
    system = first_order_system

    previous = 0.0

    for _ in range(100):
        t, _, _ = system.simulate(
            constant_input(1.0),
            duration=0.3,
            step_size=0.1,
        )

        assert t[-1] > previous
        previous = t[-1].item()
