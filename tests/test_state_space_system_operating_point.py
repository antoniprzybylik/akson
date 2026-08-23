import pytest
import torch
import warnings
from akson import StateSpaceDynamics, OperatingPoint

# First test example
@pytest.fixture
def simple_dynamics():
    A = torch.tensor([[-1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    return StateSpaceDynamics.from_linear(
        A, B, C, D, 
        u_min=torch.tensor([0.0], dtype=torch.float64), 
        u_max=torch.tensor([2.0], dtype=torch.float64)
    )

# Second test example
def _stable_siso_dynamics(**kwargs):
    # dx/dt = -x + u, y = x. Equilibrium: x = u.
    A = torch.tensor([[-1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    return StateSpaceDynamics.from_linear(A, B, C, D, **kwargs)

def test_constructor_rejects_non_1d_tensors():
    u = torch.tensor([[1.0]], dtype=torch.float64)
    x = torch.tensor([0.0], dtype=torch.float64)
    y = torch.tensor([0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="u must be 1D"):
        OperatingPoint(u, x, y)

def test_from_values_validates_equilibrium(simple_dynamics):
    # dx/dt = -x + u. Dla u=1.0, x=1.0 => dx/dt = 0.
    u = torch.tensor([1.0], dtype=torch.float64)
    x = torch.tensor([1.0], dtype=torch.float64)
    y = torch.tensor([1.0], dtype=torch.float64)
    op = OperatingPoint.from_values(u, x, y, dynamics=simple_dynamics)
    assert torch.allclose(op.x, x)

def test_from_values_rejects_non_equilibrium(simple_dynamics):
    # Dla u=1.0, x=0.0 => dx/dt = 1.0 != 0. Powinien rzucić wyjątek.
    u = torch.tensor([1.0], dtype=torch.float64)
    x = torch.tensor([0.0], dtype=torch.float64)
    y = torch.tensor([0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="is not an equilibrium"):
        OperatingPoint.from_values(u, x, y, dynamics=simple_dynamics)

def test_from_values_warns_on_y_mismatch(simple_dynamics):
    # Poprawne x i u, ale błędne y (powinno być 1.0, dajemy 5.0)
    u = torch.tensor([1.0], dtype=torch.float64)
    x = torch.tensor([1.0], dtype=torch.float64)
    y = torch.tensor([5.0], dtype=torch.float64)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        OperatingPoint.from_values(u, x, y, dynamics=simple_dynamics)
        assert any("does not match g(x, u)" in str(warn.message) for warn in w)

def test_from_input_finds_equilibrium(simple_dynamics):
    u = torch.tensor([1.5], dtype=torch.float64)
    op = OperatingPoint.from_input(simple_dynamics, u)
    assert torch.allclose(op.x, torch.tensor([1.5], dtype=torch.float64), atol=1e-6)
    assert torch.allclose(op.u, u)

def test_from_input_fails_on_unstable_system():
    # dx/dt = x + u (układ niestabilny, brak punktu równowagi dla u != 0)
    A = torch.tensor([[1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    dynamics = StateSpaceDynamics.from_linear(
        A, B, C, D, 
        skip_controllability_check=True, 
        skip_observability_check=True
    )
    
    u = torch.tensor([1.0], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="Could not find a stable equilibrium"):
        OperatingPoint.from_input(dynamics, u, max_attempts=3, initial_duration=0.1)


def test_operating_point_rejects_non_1d_u():
    with pytest.raises(ValueError, match="u must be 1D"):
        OperatingPoint(
            u=torch.zeros(1, 1, dtype=torch.float64),
            x=torch.zeros(1, dtype=torch.float64),
            y=torch.zeros(1, dtype=torch.float64),
        )


def test_operating_point_rejects_non_1d_x():
    with pytest.raises(ValueError, match="x must be 1D"):
        OperatingPoint(
            u=torch.zeros(1, dtype=torch.float64),
            x=torch.zeros(1, 1, dtype=torch.float64),
            y=torch.zeros(1, dtype=torch.float64),
        )


def test_operating_point_rejects_non_1d_y():
    with pytest.raises(ValueError, match="y must be 1D"):
        OperatingPoint(
            u=torch.zeros(1, dtype=torch.float64),
            x=torch.zeros(1, dtype=torch.float64),
            y=torch.zeros(1, 1, dtype=torch.float64),
        )


def test_operating_point_stores_clones_not_references():
    u = torch.tensor([1.0], dtype=torch.float64)
    op = OperatingPoint(u=u, x=torch.zeros(1, dtype=torch.float64), y=torch.zeros(1, dtype=torch.float64))
    u += 100.0
    assert op.u.item() == pytest.approx(1.0)


def test_operating_point_repr_contains_values():
    op = OperatingPoint(
        u=torch.tensor([1.5], dtype=torch.float64),
        x=torch.tensor([0.0], dtype=torch.float64),
        y=torch.tensor([2.5], dtype=torch.float64),
    )
    text = repr(op)
    assert "1.5" in text
    assert "2.5" in text


def test_from_values_without_dynamics_performs_no_validation():
    # With dynamics=None the equilibrium/shape/bound checks are skipped
    # entirely, so an x that would not actually be an equilibrium is
    # accepted as-is.
    op = OperatingPoint.from_values(
        u=torch.tensor([1.0], dtype=torch.float64),
        x=torch.tensor([999.0], dtype=torch.float64),
        y=torch.tensor([999.0], dtype=torch.float64),
        dynamics=None,
    )
    assert op.x.item() == pytest.approx(999.0)


def test_from_values_accepts_true_equilibrium():
    dynamics = _stable_siso_dynamics()
    op = OperatingPoint.from_values(
        u=torch.tensor([2.0], dtype=torch.float64),
        x=torch.tensor([2.0], dtype=torch.float64),
        y=torch.tensor([2.0], dtype=torch.float64),
        dynamics=dynamics,
    )
    assert op.x.item() == pytest.approx(2.0)


def test_from_values_rejects_non_equilibrium():
    dynamics = _stable_siso_dynamics()
    with pytest.raises(ValueError, match="not an equilibrium"):
        OperatingPoint.from_values(
            u=torch.tensor([2.0], dtype=torch.float64),
            x=torch.tensor([0.0], dtype=torch.float64),  # x != u, so dx/dt != 0
            y=torch.tensor([0.0], dtype=torch.float64),
            dynamics=dynamics,
        )


def test_from_values_warns_on_y_mismatch():
    dynamics = _stable_siso_dynamics()
    with pytest.warns(RuntimeWarning, match="does not match"):
        OperatingPoint.from_values(
            u=torch.tensor([2.0], dtype=torch.float64),
            x=torch.tensor([2.0], dtype=torch.float64),
            y=torch.tensor([999.0], dtype=torch.float64),  # g(x, u) = x = 2, not 999
            dynamics=dynamics,
        )


def test_from_values_validates_u_shape_against_dynamics():
    dynamics = _stable_siso_dynamics()
    with pytest.raises(ValueError, match="u has bad shape"):
        OperatingPoint.from_values(
            u=torch.tensor([1.0, 2.0], dtype=torch.float64),
            x=torch.tensor([2.0], dtype=torch.float64),
            y=torch.tensor([2.0], dtype=torch.float64),
            dynamics=dynamics,
        )


def test_from_values_validates_x_shape_against_dynamics():
    dynamics = _stable_siso_dynamics()
    with pytest.raises(ValueError, match="x has bad shape"):
        OperatingPoint.from_values(
            u=torch.tensor([2.0], dtype=torch.float64),
            x=torch.tensor([2.0, 3.0], dtype=torch.float64),
            y=torch.tensor([2.0], dtype=torch.float64),
            dynamics=dynamics,
        )


def test_from_values_validates_y_shape_against_dynamics():
    dynamics = _stable_siso_dynamics()
    with pytest.raises(ValueError, match="y has bad shape"):
        OperatingPoint.from_values(
            u=torch.tensor([2.0], dtype=torch.float64),
            x=torch.tensor([2.0], dtype=torch.float64),
            y=torch.tensor([2.0, 3.0], dtype=torch.float64),
            dynamics=dynamics,
        )


def test_from_values_enforces_u_bounds():
    dynamics = _stable_siso_dynamics(u_min=torch.tensor([0.0]), u_max=torch.tensor([1.0]))
    with pytest.raises(ValueError, match="exceeds u_max"):
        OperatingPoint.from_values(
            u=torch.tensor([2.0], dtype=torch.float64),
            x=torch.tensor([2.0], dtype=torch.float64),
            y=torch.tensor([2.0], dtype=torch.float64),
            dynamics=dynamics,
        )


def test_from_input_finds_correct_equilibrium():
    dynamics = _stable_siso_dynamics()
    op = OperatingPoint.from_input(dynamics, torch.tensor([3.0], dtype=torch.float64))
    # Equilibrium of dx/dt = -x + u is x = u = 3
    assert op.x.item() == pytest.approx(3.0, abs=1e-4)
    assert op.y.item() == pytest.approx(3.0, abs=1e-4)


def test_from_input_rejects_wrong_u_shape():
    dynamics = _stable_siso_dynamics()
    with pytest.raises(ValueError, match="u has bad shape"):
        OperatingPoint.from_input(dynamics, torch.tensor([1.0, 2.0], dtype=torch.float64))


def test_from_input_validates_u_against_dynamics_bounds():
    dynamics = _stable_siso_dynamics(u_min=torch.tensor([0.0]), u_max=torch.tensor([1.0]))
    with pytest.raises(ValueError, match="exceeds u_max"):
        OperatingPoint.from_input(dynamics, torch.tensor([5.0], dtype=torch.float64))


def test_from_input_uses_custom_initial_state():
    dynamics = _stable_siso_dynamics()
    op = OperatingPoint.from_input(
        dynamics, torch.tensor([1.0], dtype=torch.float64),
        x_init=torch.tensor([0.9], dtype=torch.float64),
    )
    assert op.x.item() == pytest.approx(1.0, abs=1e-4)


def test_from_input_rejects_wrong_x_init_shape():
    dynamics = _stable_siso_dynamics()
    with pytest.raises(ValueError, match="x has bad shape"):
        OperatingPoint.from_input(
            dynamics, torch.tensor([1.0], dtype=torch.float64),
            x_init=torch.tensor([1.0, 2.0], dtype=torch.float64),
        )
