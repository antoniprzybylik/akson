import torch
import torch.nn as nn
import matplotlib.pyplot as pyplot
from akson import StateSpaceDynamics
from data_state_space_system_step_response_lti import (
    matlab_continous_t_test1,
    matlab_continous_y_test1,
    matlab_y_step_tustin_method_test1,
    matlab_continous_t_test2,
    matlab_continous_y_test2,
    matlab_y_step_zoh_method_test2,
)


def create_siso_lti_state_space_system():
    device = torch.device("cpu")
    dtype = torch.float64

    A = torch.tensor([[-1.0, 2.0], [-3.0, 0.0]], device=device, dtype=dtype)
    B = torch.tensor([[1.0], [2.0]], device=device, dtype=dtype)
    C = torch.tensor([[-1.0, 0.1]], device=device, dtype=dtype)
    D = torch.tensor([[0.0]], device=device, dtype=dtype)
    initial_state = torch.tensor([[0.0], [0.0]], device=device, dtype=dtype)

    return StateSpaceDynamics.from_linear(A, B, C, D)


def test_discrete_step_response_siso():
    system_dynamics = create_siso_lti_state_space_system()

    dt = 0.1
    duration = 11.4
    t_step, y_step = system_dynamics.discrete_step_response(duration=duration, dt=dt)

    assert torch.allclose(t_step, torch.arange(0.0, 11.5, 0.1, dtype=torch.float64))

    # Discrete step responses don't need to match precisely as we use
    # different discretization methods
    assert (
        torch.abs(y_step.squeeze() - matlab_y_step_tustin_method_test1.squeeze()) < 0.07
    ).all()


def _binsearch_le(tab, x):
    start = 0
    end = len(tab) - 1
    while start < end:
        m = (start + end + 1) // 2
        if tab[m] <= x:
            start = m
        else:
            end = m - 1
    return start


def _exact_step_response_test1(t: float):
    if t < 0:
        raise ValueError("Bad time")
    if t < matlab_continous_t_test1[0]:
        return matlab_continous_y_test1[0] * t / matlab_continous_t_test1[0]
    if t >= matlab_continous_t_test1[-1]:
        return matlab_continous_y_test1[-1]

    i = _binsearch_le(matlab_continous_t_test1, t)
    ratio = (t - matlab_continous_t_test1[i]) / (
        matlab_continous_t_test1[i + 1] - matlab_continous_t_test1[i]
    )

    return (
        matlab_continous_y_test1[i] * (1.0 - ratio)
        + matlab_continous_y_test1[i + 1] * ratio
    )


def test_continous_step_response_siso():
    system_dynamics = create_siso_lti_state_space_system()

    duration = 10.0
    t_step, y_step = system_dynamics.step_response(duration=duration)

    for i in range(t_step.shape[0]):
        assert (
            torch.abs(_exact_step_response_test1(t_step[i]) - y_step[i]).item() < 0.045
        )


def create_mimo_lti_state_space_system():
    device = torch.device("cpu")
    dtype = torch.float64

    A = torch.tensor([[-1.0, 2.0], [-3.0, 0.0]], device=device, dtype=dtype)
    B = torch.tensor([[1.0, 0.1], [2.0, -1.0]], device=device, dtype=dtype)
    C = torch.tensor([[-1.0, 0.1], [0.0, -1.0]], device=device, dtype=dtype)
    D = torch.tensor([[0.0, 0.0], [0.0, 0.0]], device=device, dtype=dtype)
    initial_state = torch.tensor([[0.0], [0.0]], device=device, dtype=dtype)

    return StateSpaceDynamics.from_linear(A, B, C, D)


def test_discrete_step_response_mimo():
    system_dynamics = create_mimo_lti_state_space_system()

    dt = 0.1
    duration = 13.8
    t_step, y_step = system_dynamics.discrete_step_response(duration=duration, dt=dt)

    assert torch.allclose(t_step, torch.arange(0.0, 13.9, 0.1, dtype=torch.float64))

    # Discrete step responses don't need to match precisely as we use
    # different discretization methods
    print(y_step.shape, matlab_y_step_zoh_method_test2.shape)
    assert (torch.abs(y_step - matlab_y_step_zoh_method_test2) < 5e-5).all()


def _exact_step_response_test2(t: float):
    if t < 0:
        raise ValueError("Bad time")
    if t < matlab_continous_t_test2[0]:
        return matlab_continous_y_test2[0] * t / matlab_continous_t_test2[0]
    if t >= matlab_continous_t_test2[-1]:
        return matlab_continous_y_test2[-1]

    i = _binsearch_le(matlab_continous_t_test2, t)
    ratio = (t - matlab_continous_t_test2[i]) / (
        matlab_continous_t_test2[i + 1] - matlab_continous_t_test2[i]
    )

    return (
        matlab_continous_y_test2[i] * (1.0 - ratio)
        + matlab_continous_y_test2[i + 1] * ratio
    )


def test_continous_step_response_mimo():
    system_dynamics = create_mimo_lti_state_space_system()

    duration = 13.723407154238418
    t_step, y_step = system_dynamics.step_response(duration=duration)
    print(t_step.shape, y_step.shape)

    for i in range(t_step.shape[0]):
        assert (
            torch.abs(_exact_step_response_test2(t_step[i]) - y_step[i]).max().item()
            < 0.055
        )
