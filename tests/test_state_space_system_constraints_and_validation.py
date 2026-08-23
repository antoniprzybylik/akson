import pytest
import torch
import torch.nn as nn
import sympy as sp
import warnings
from akson import StateSpaceDynamics, StateSpaceSystem

def test_from_linear_rejects_complex_matrices():
    A = torch.tensor([[1.0 + 1j]], dtype=torch.complex64)
    B = torch.tensor([[1.0]], dtype=torch.complex64)
    C = torch.tensor([[1.0]], dtype=torch.complex64)
    D = torch.tensor([[0.0]], dtype=torch.complex64)
    with pytest.raises(ValueError, match="Complex state-space matrices are not supported"):
        StateSpaceDynamics.from_linear(A, B, C, D)

def test_from_linear_rejects_bad_shapes():
    A = torch.tensor([[-1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0]], dtype=torch.float64) # Bad shape, should be (2, m)
    C = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="B must be"):
        StateSpaceDynamics.from_linear(A, B, C, D)

def test_from_linear_checks_controllability():
    A = torch.tensor([[-1.0, 0.0], [0.0, -2.0]], dtype=torch.float64)
    B = torch.tensor([[1.0], [0.0]], dtype=torch.float64) # Not controllable
    C = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="System is not controllable"):
        StateSpaceDynamics.from_linear(A, B, C, D)

def test_from_linear_checks_observability():
    A = torch.tensor([[-1.0, 0.0], [0.0, -2.0]], dtype=torch.float64)
    B = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0, 0.0]], dtype=torch.float64) # Not observable
    D = torch.tensor([[0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="System is not observable"):
        StateSpaceDynamics.from_linear(A, B, C, D)

def test_from_linear_skip_checks():
    A = torch.tensor([[-1.0, 0.0], [0.0, -2.0]], dtype=torch.float64)
    B = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    C = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    # Should not throw exception as the checks are disabled
    dynamics = StateSpaceDynamics.from_linear(
        A, B, C, D, 
        skip_controllability_check=True, 
        skip_observability_check=True
    )
    assert dynamics.is_lti

def test_u_min_max_validation_during_simulation():
    A = torch.tensor([[-1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    
    dynamics = StateSpaceDynamics.from_linear(
        A, B, C, D, 
        u_min=torch.tensor([0.0], dtype=torch.float64),
        u_max=torch.tensor([1.0], dtype=torch.float64)
    )
    system = StateSpaceSystem(dynamics)
    
    def bad_u(t):
        return torch.tensor([2.0], dtype=torch.float64) # Exceeds u_max
        
    with pytest.raises(ValueError, match="Input \\[2.0\\] exceeds u_max \\[1.0\\]."):
        system.simulate(bad_u, duration=1.0, step_size=0.1)

def test_x_min_max_validation_during_simulation():
    A = torch.tensor([[1.0]], dtype=torch.float64) # Unstable, we want the state to grow fast
    B = torch.tensor([[1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    
    dynamics = StateSpaceDynamics.from_linear(
        A, B, C, D,
        x_max=torch.tensor([0.5], dtype=torch.float64),
        skip_controllability_check=True,
        skip_observability_check=True
    )
    system = StateSpaceSystem(dynamics)
    
    def u(t):
        return torch.tensor([1.0], dtype=torch.float64)
        
    with pytest.raises(ValueError, match="State exceeded limits"):
        system.simulate(u, duration=2.0, step_size=0.1)

class BadFModule(nn.Module):
    def forward(self, t, x, u):
        return torch.tensor([1.0, 2.0], dtype=torch.float64) # Bad shape (state_size=1)

class BadGModule(nn.Module):
    def forward(self, x, u):
        return torch.tensor([1.0, 2.0], dtype=torch.float64) # Bad shape (n_outputs=1)

class GoodFModule(nn.Module):
    def forward(self, t, x, u):
        return torch.tensor([1.0], dtype=torch.float64)

class GoodGModule(nn.Module):
    def forward(self, x, u):
        return torch.tensor([1.0], dtype=torch.float64)

def test_f_module_output_shape_validation():
    with pytest.raises(ValueError, match="f must return shape"):
        StateSpaceDynamics(
            BadFModule(), GoodGModule(), 
            n_inputs=1, n_outputs=1, state_size=1
        )

def test_g_module_output_shape_validation():
    with pytest.raises(ValueError, match="g must return shape"):
        StateSpaceDynamics(
            GoodFModule(), BadGModule(), 
            n_inputs=1, n_outputs=1, state_size=1
        )

def test_from_linear_rejects_mixed_dtypes():
    A = torch.tensor([[-1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0]], dtype=torch.float32)  # Dtype differs from the rest
    C = torch.tensor([[1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    
    with pytest.raises(ValueError, match="same datatype and device"):
        StateSpaceDynamics.from_linear(A, B, C, D)


def test_from_linear_rejects_non_square_A():
    A = torch.tensor([[-1.0, 0.0]], dtype=torch.float64)  # 1x2 instead of 2x2
    B = torch.tensor([[1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    
    with pytest.raises(ValueError, match="A must be square"):
        StateSpaceDynamics.from_linear(A, B, C, D)


def test_from_linear_rejects_bad_B_shape():
    A = torch.tensor([[-1.0, 0.0], [0.0, -2.0]], dtype=torch.float64)
    B = torch.tensor([[1.0, 0.0]], dtype=torch.float64)  # Bad shape
    C = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    
    with pytest.raises(ValueError, match="B must be"):
        StateSpaceDynamics.from_linear(A, B, C, D)


def test_from_linear_rejects_bad_C_shape():
    A = torch.tensor([[-1.0, 0.0], [0.0, -2.0]], dtype=torch.float64)
    B = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0]], dtype=torch.float64)  # Bad shape
    D = torch.tensor([[0.0]], dtype=torch.float64)
    
    with pytest.raises(ValueError, match="C must be"):
        StateSpaceDynamics.from_linear(A, B, C, D)


def test_from_linear_rejects_bad_D_shape():
    A = torch.tensor([[-1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0, 0.0]], dtype=torch.float64)  # Bad shape
    
    with pytest.raises(ValueError, match="D must be"):
        StateSpaceDynamics.from_linear(A, B, C, D)


def test_constructor_rejects_u_min_greater_than_u_max():
    class F(nn.Module):
        def forward(self, t, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    class G(nn.Module):
        def forward(self, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="u_min must be <= u_max"):
        StateSpaceDynamics(
            F(), G(),
            n_inputs=1, n_outputs=1, state_size=1,
            u_min=torch.tensor([2.0], dtype=torch.float64),
            u_max=torch.tensor([1.0], dtype=torch.float64)
        )


def test_constructor_rejects_x_min_greater_than_x_max():
    class F(nn.Module):
        def forward(self, t, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    class G(nn.Module):
        def forward(self, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="x_min must be <= x_max"):
        StateSpaceDynamics(
            F(), G(),
            n_inputs=1, n_outputs=1, state_size=1,
            x_min=torch.tensor([2.0], dtype=torch.float64),
            x_max=torch.tensor([1.0], dtype=torch.float64)
        )


def test_constructor_rejects_y_min_greater_than_y_max():
    class F(nn.Module):
        def forward(self, t, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    class G(nn.Module):
        def forward(self, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="y_min must be <= y_max"):
        StateSpaceDynamics(
            F(), G(),
            n_inputs=1, n_outputs=1, state_size=1,
            y_min=torch.tensor([2.0], dtype=torch.float64),
            y_max=torch.tensor([1.0], dtype=torch.float64)
        )


def test_constructor_rejects_bad_u_min_shape():
    class F(nn.Module):
        def forward(self, t, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    class G(nn.Module):
        def forward(self, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="u_min has bad shape"):
        StateSpaceDynamics(
            F(), G(),
            n_inputs=1, n_outputs=1, state_size=1,
            u_min=torch.tensor([1.0, 2.0], dtype=torch.float64)
        )


def test_constructor_rejects_bad_x_min_shape():
    class F(nn.Module):
        def forward(self, t, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    class G(nn.Module):
        def forward(self, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="x_min has bad shape"):
        StateSpaceDynamics(
            F(), G(),
            n_inputs=1, n_outputs=1, state_size=1,
            x_min=torch.tensor([1.0, 2.0], dtype=torch.float64)
        )


def test_constructor_rejects_bad_y_min_shape():
    class F(nn.Module):
        def forward(self, t, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    class G(nn.Module):
        def forward(self, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="y_min has bad shape"):
        StateSpaceDynamics(
            F(), G(),
            n_inputs=1, n_outputs=1, state_size=1,
            y_min=torch.tensor([1.0, 2.0], dtype=torch.float64)
        )


def test_constructor_rejects_non_positive_n_inputs():
    class F(nn.Module):
        def forward(self, t, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    class G(nn.Module):
        def forward(self, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="n_inputs must be positive"):
        StateSpaceDynamics(
            F(), G(),
            n_inputs=0, n_outputs=1, state_size=1
        )


def test_constructor_rejects_non_positive_n_outputs():
    class F(nn.Module):
        def forward(self, t, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    class G(nn.Module):
        def forward(self, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="n_outputs must be positive"):
        StateSpaceDynamics(
            F(), G(),
            n_inputs=1, n_outputs=0, state_size=1
        )


def test_constructor_rejects_non_positive_state_size():
    class F(nn.Module):
        def forward(self, t, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    class G(nn.Module):
        def forward(self, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="state_size must be positive"):
        StateSpaceDynamics(
            F(), G(),
            n_inputs=1, n_outputs=1, state_size=0
        )


def test_simulate_without_initial_state_or_operating_point_raises():
    class F(nn.Module):
        def forward(self, t, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    class G(nn.Module):
        def forward(self, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    dynamics = StateSpaceDynamics(
        F(), G(),
        n_inputs=1, n_outputs=1, state_size=1
    )
    
    def u(t):
        return torch.tensor([1.0], dtype=torch.float64)
    
    with pytest.raises(RuntimeError, match="Initial system state"):
        dynamics.simulate(u, duration=1.0)


def test_state_space_system_without_initial_state_or_operating_point_raises():
    class F(nn.Module):
        def forward(self, t, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    class G(nn.Module):
        def forward(self, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    dynamics = StateSpaceDynamics(
        F(), G(),
        n_inputs=1, n_outputs=1, state_size=1
    )
    
    with pytest.raises(RuntimeError, match="System state x not provided"):
        StateSpaceSystem(dynamics)


def test_state_space_system_reset_without_state_raises():
    class F(nn.Module):
        def forward(self, t, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    class G(nn.Module):
        def forward(self, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    dynamics = StateSpaceDynamics(
        F(), G(),
        n_inputs=1, n_outputs=1, state_size=1
    )
    
    # Set the initial state
    system = StateSpaceSystem(dynamics, x=torch.tensor([1.0], dtype=torch.float64))
    
    # Get rid of the operating point
    dynamics.operating_point = None
    
    with pytest.raises(RuntimeError, match="System state x not provided"):
        system.reset()


def test_state_space_system_rejects_bad_x_shape():
    A = torch.tensor([[-1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    
    dynamics = StateSpaceDynamics.from_linear(A, B, C, D)
    
    with pytest.raises(ValueError, match="must have shape"):
        StateSpaceSystem(dynamics, x=torch.tensor([1.0, 2.0], dtype=torch.float64))


def test_simulate_rejects_bad_initial_state_shape():
    A = torch.tensor([[-1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    
    dynamics = StateSpaceDynamics.from_linear(A, B, C, D)
    
    def u(t):
        return torch.tensor([1.0], dtype=torch.float64)
    
    with pytest.raises(ValueError, match="initial_state must have shape"):
        dynamics.simulate(
            u, 
            duration=1.0, 
            initial_state=torch.tensor([1.0, 2.0], dtype=torch.float64)
        )


def test_simulate_rejects_bad_u_func_return_shape():
    A = torch.tensor([[-1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    
    dynamics = StateSpaceDynamics.from_linear(A, B, C, D)
    
    def bad_u(t):
        return torch.tensor([1.0, 2.0], dtype=torch.float64)
    
    with pytest.raises(ValueError, match="u_func must return shape"):
        dynamics.simulate(bad_u, duration=1.0)


def test_from_tf_rejects_empty_matrix():
    s = sp.var('s')
    H = sp.Matrix([])  # Empty matrix
    
    with pytest.raises(ValueError, match="non-empty matrix"):
        StateSpaceDynamics.from_tf(H, s)


def test_from_tf_warns_on_float_entries():
    s = sp.var('s')
    H = sp.Matrix([[1.5/(s+1)]])  # Float instead of Rational
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        StateSpaceDynamics.from_tf(H, s)
        
        assert any("Float entries" in str(warn.message) for warn in w)


def test_from_tf_warns_on_large_coefficients():
    s = sp.var('s')
    H = sp.Matrix([[2000000/(s+1)]])  # Big coefficient
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        StateSpaceDynamics.from_tf(H, s)
        
        assert any("Large coefficients" in str(warn.message) for warn in w)


def test_warns_when_max_spectral_radius_supplied_for_nonlinear():
    class F(nn.Module):
        def forward(self, t, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    class G(nn.Module):
        def forward(self, x, u):
            return torch.tensor([1.0], dtype=torch.float64)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        StateSpaceDynamics(
            F(), G(),
            n_inputs=1, n_outputs=1, state_size=1,
            max_spectral_radius=1.0
        )
        
        assert any("max_spectral_radius was supplied" in str(warn.message) for warn in w)
