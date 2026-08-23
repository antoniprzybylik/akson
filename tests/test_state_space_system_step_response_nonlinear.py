import torch
import torch.nn as nn
import matplotlib.pyplot as pyplot
from akson import StateSpaceDynamics, OperatingPoint
from data_state_space_system_step_response_nonlinear import (
    matlab_continous_t,
    matlab_continous_y,
    matlab_y_step_tustin_method,
)


def create_siso_nonlinear_state_space_system():
    device = torch.device("cpu")
    dtype = torch.float64

    A = torch.tensor(
        [[-0.0333, 0.0667], [-0.0800, -0.4000]], device=device, dtype=dtype
    )
    B = torch.tensor([[0.0], [0.2]], device=device, dtype=dtype)
    C = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
    D = torch.tensor([[0.0]], device=device, dtype=dtype)

    class FModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("A", A)
            self.register_buffer("B", B.squeeze(-1))

        def forward(self, t, x, u):
            return self.A @ x + self.B * u

    class GModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("C", C.squeeze(0))
            self.register_buffer("D", D.squeeze())

            self.nonlinear_block = lambda x: (
                x
                - 1.75
                * (
                    (torch.exp(0.5 * x) - torch.exp(-0.5 * x))
                    / (torch.exp(0.5 * x) + torch.exp(-0.5 * x))
                )
            )

        def forward(self, x, u):
            return self.nonlinear_block(self.C @ x + self.D * u)

    system_dynamics = StateSpaceDynamics(
        FModule(),
        GModule(),
        n_inputs=1,
        n_outputs=1,
        state_size=2,
        dtype=dtype,
        device=device,
    )

    op = OperatingPoint.from_input(
        system_dynamics, torch.zeros((1,), dtype=dtype, device=device)
    )
    system_dynamics.operating_point = op

    return system_dynamics


def test_discrete_step_response():
    system_dynamics = create_siso_nonlinear_state_space_system()

    dt = 5.0
    duration = 125.0
    t_step, y_step = system_dynamics.discrete_step_response(duration=duration, dt=dt)

    assert torch.allclose(t_step, torch.arange(0.0, 130.0, 5.0, dtype=torch.float64))

    # Discrete step responses don't need to match precisely as we use
    # different discretization methods
    assert (
        torch.abs(y_step.squeeze() - matlab_y_step_tustin_method.squeeze()) < 0.0064
    ).all()


def binsearch_le(tab, x):
    start = 0
    end = len(tab) - 1
    while start < end:
        m = (start + end + 1) // 2
        if tab[m] <= x:
            start = m
        else:
            end = m - 1
    return start


def exact_step_response(t: float):
    if t < 0:
        raise ValueError("Bad time")
    if t < matlab_continous_t[0]:
        return matlab_continous_y[0] * t / matlab_continous_t[0]
    if t >= matlab_continous_t[-1]:
        return matlab_continous_y[-1]

    i = binsearch_le(matlab_continous_t, t)
    ratio = (t - matlab_continous_t[i]) / (
        matlab_continous_t[i + 1] - matlab_continous_t[i]
    )

    return matlab_continous_y[i] * (1.0 - ratio) + matlab_continous_y[i + 1] * ratio


def test_continous_step_response():
    system_dynamics = create_siso_nonlinear_state_space_system()

    duration = 125.0
    t_step, y_step = system_dynamics.step_response(duration=duration)

    for i in range(t_step.shape[0]):
        assert torch.abs(exact_step_response(t_step[i]) - y_step[i]).item() < 1e-4
