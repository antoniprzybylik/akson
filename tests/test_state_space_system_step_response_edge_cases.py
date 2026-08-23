import pytest
import torch
import mini_ode
from akson import StateSpaceDynamics, OperatingPoint

@pytest.fixture
def lti_dynamics():
    A = torch.tensor([[-1.0]], dtype=torch.float64)
    B = torch.tensor([[1.0]], dtype=torch.float64)
    C = torch.tensor([[1.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    return StateSpaceDynamics.from_linear(A, B, C, D)

def test_step_response_no_operating_point_raises():
    dynamics = StateSpaceDynamics.from_linear(
        torch.tensor([[-1.0]], dtype=torch.float64), 
        torch.tensor([[1.0]], dtype=torch.float64), 
        torch.tensor([[1.0]], dtype=torch.float64), 
        torch.tensor([[0.0]], dtype=torch.float64)
    )
    dynamics.operating_point = None
    
    with pytest.raises(RuntimeError, match="requires a known operating point"):
        dynamics.step_response(duration=1.0)

def test_step_response_two_operating_points_raises(lti_dynamics):
    op = OperatingPoint.from_values(
        torch.tensor([0.0], dtype=torch.float64), 
        torch.tensor([0.0], dtype=torch.float64), 
        torch.tensor([0.0], dtype=torch.float64),
        dynamics=lti_dynamics
    )
    lti_dynamics.operating_point = op
    
    with pytest.raises(RuntimeError, match="Two operating points were provided"):
        lti_dynamics.step_response(duration=1.0, operating_point=op)

def test_step_response_zero_amp_raises(lti_dynamics):
    op = OperatingPoint.from_values(
        torch.tensor([0.0], dtype=torch.float64), 
        torch.tensor([0.0], dtype=torch.float64), 
        torch.tensor([0.0], dtype=torch.float64),
        dynamics=lti_dynamics
    )
    lti_dynamics.operating_point = op
    
    with pytest.raises(ValueError, match="amp must not be zero"):
        lti_dynamics.step_response(duration=1.0, amp=0.0)

def test_discrete_step_response_mismatched_dt_raises(lti_dynamics):
    op = OperatingPoint.from_values(
        torch.tensor([0.0], dtype=torch.float64), 
        torch.tensor([0.0], dtype=torch.float64), 
        torch.tensor([0.0], dtype=torch.float64),
        dynamics=lti_dynamics
    )
    lti_dynamics.operating_point = op
    
    # Force solver with step=0.5, but demand dt=0.1 at the same time
    custom_solver = mini_ode.RK4MethodSolver(step=0.5)
    
    with pytest.raises(ValueError, match="does not match the provided dt"):
        lti_dynamics.discrete_step_response(duration=1.0, dt=0.1, solver=custom_solver)
