import pytest
import torch
import torch.nn as nn
from akson import StateSpaceDynamics, StateSpaceSystem, OperatingPoint


@pytest.fixture
def lti_dynamics():
    A = torch.tensor([[-1.0, 0.0], [0.0, -2.0]], dtype=torch.float64)
    B = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    return StateSpaceDynamics.from_linear(
        A, B, C, D, skip_controllability_check=True, skip_observability_check=True
    )


@pytest.fixture
def nonlinear_dynamics():
    class F(nn.Module):
        def forward(self, t, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    class G(nn.Module):
        def forward(self, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    return StateSpaceDynamics(F(), G(), n_inputs=1, n_outputs=1, state_size=1)


def test_operating_point_repr():
    u = torch.tensor([1.0], dtype=torch.float64)
    x = torch.tensor([2.0], dtype=torch.float64)
    y = torch.tensor([3.0], dtype=torch.float64)
    op = OperatingPoint(u, x, y)

    repr_str = repr(op)
    assert "OperatingPoint" in repr_str
    assert "u=" in repr_str
    assert "x=" in repr_str
    assert "y=" in repr_str
    assert "1.0" in repr_str
    assert "2.0" in repr_str
    assert "3.0" in repr_str


def test_lti_repr_latex(lti_dynamics):
    latex = lti_dynamics._repr_latex_()

    # Should contain LaTeX-formatted matrices
    assert r"\begin{gather}" in latex
    assert r"\dot{x}" in latex
    assert r"\begin{bmatrix}" in latex
    assert r"\end{bmatrix}" in latex
    assert "x" in latex
    assert "u" in latex
    assert "y" in latex


def test_nonlinear_repr_latex(nonlinear_dynamics):
    latex = nonlinear_dynamics._repr_latex_()

    # Should contain general state space system equation (with F and G)
    assert r"\begin{gather}" in latex
    assert r"\dot{x}" in latex
    assert "F(t, x, u)" in latex
    assert "G(t, x, u)" in latex


def test_state_space_system_repr(lti_dynamics):
    system = StateSpaceSystem(lti_dynamics)
    repr_str = repr(system)

    assert "StateSpaceSystem" in repr_str
    assert "dynamics=" in repr_str
    assert "x=" in repr_str


def test_state_space_system_repr_custom_initial_state(lti_dynamics):
    custom_x = torch.tensor([5.0, 6.0], dtype=torch.float64)
    system = StateSpaceSystem(lti_dynamics, x=custom_x)
    repr_str = repr(system)

    assert "StateSpaceSystem" in repr_str
    assert "x=" in repr_str
