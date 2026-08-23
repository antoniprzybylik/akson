import torch
import sympy as sp
from akson import StateSpaceDynamics


def test_conversion_and_simulation_siso_system_1():
    # Directly construct state space dynamics from transfer function
    s = sp.symbols("s")
    H = sp.Matrix([[(s**2 + s - 3) / (s**3 + s**2 + 2 * s + 1)]])
    system_dynamics1 = StateSpaceDynamics.from_tf(H, s)

    # Construct state space dynamics from LTI system matrices calculated with MatLab
    A = torch.tensor(
        [[-1.0, -2.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    B = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)
    C = torch.tensor([[1.0, 1.0, -3.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    system_dynamics2 = StateSpaceDynamics.from_linear(A, B, C, D)

    t1, y1 = system_dynamics1.step_response(step_size=0.01, duration=1.0)
    t2, y2 = system_dynamics2.step_response(step_size=0.01, duration=1.0)

    assert torch.allclose(t1, t2)
    assert torch.allclose(y1, y2)


def test_conversion_and_simulation_siso_system_2():
    # Directly construct state space dynamics from transfer function
    s = sp.symbols("s")
    H = sp.Matrix([[(s**3 + s**2 + s - 3) / (s + 1) ** 4]])
    system_dynamics1 = StateSpaceDynamics.from_tf(H, s)

    # Construct state space dynamics from LTI system matrices calculated with MatLab
    A = torch.tensor(
        [
            [-4.0, -6.0, -4.0, -1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    B = torch.tensor([[1.0], [0.0], [0.0], [0.0]], dtype=torch.float64)
    C = torch.tensor([[1.0, 1.0, 1.0, -3.0]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    system_dynamics2 = StateSpaceDynamics.from_linear(A, B, C, D)

    t1, y1 = system_dynamics1.step_response(step_size=0.01, duration=1.0)
    t2, y2 = system_dynamics2.step_response(step_size=0.01, duration=1.0)

    assert torch.allclose(t1, t2)
    assert torch.allclose(y1, y2)


def test_conversion_and_simulation_siso_system_3():
    # Directly construct state space dynamics from transfer function
    s = sp.symbols("s")
    H = sp.Matrix([[10 / (750 * s**2 + 325 * s + 14)]])
    system_dynamics1 = StateSpaceDynamics.from_tf(H, s)

    # Construct state space dynamics from LTI system matrices calculated with MatLab
    A = torch.tensor(
        [[-0.433333333333333, -0.018666666666667], [1.0, 0.0]], dtype=torch.float64
    )
    B = torch.tensor(
        [
            [1.0],
            [0.0],
        ],
        dtype=torch.float64,
    )
    C = torch.tensor([[0.0, 0.013333333333333]], dtype=torch.float64)
    D = torch.tensor([[0.0]], dtype=torch.float64)
    system_dynamics2 = StateSpaceDynamics.from_linear(A, B, C, D)

    t1, y1 = system_dynamics1.step_response(step_size=0.01, duration=1.0)
    t2, y2 = system_dynamics2.step_response(step_size=0.01, duration=1.0)

    assert torch.allclose(t1, t2)
    assert torch.allclose(y1, y2)


def test_conversion_and_simulation_mimo_system():
    # Directly construct state space dynamics from transfer function
    s = sp.symbols("s")
    H = sp.Matrix(
        [
            [(s**2 + s - 3) / (s**3 + s**2 + 2 * s + 1), 1 / s, s / (s**2 + 2 * s + 1)],
            [
                (s + 3) / (s**2 + 2 * s + 1),
                1 / (s**2 + 2 * s),
                (s - 7) / (s**2 + 3 * s + 2),
            ],
        ]
    )
    system_dynamics1 = StateSpaceDynamics.from_tf(H, s)

    # Construct state space dynamics from LTI system matrices calculated with MatLab
    A = torch.tensor(
        [
            [-1, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        ],
        dtype=torch.float64,
    )
    B = torch.tensor(
        [
            [2, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [2, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 4],
            [0, 0, 0],
        ],
        dtype=torch.float64,
    )
    C = torch.tensor(
        [
            [0.5, 0.5, -1.5, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 1.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.25, -1.75],
        ],
        dtype=torch.float64,
    )
    D = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=torch.float64,
    )
    # We demand controllability and observability from our realization but not from MatLAB. Our implementation is better.
    system_dynamics2 = StateSpaceDynamics.from_linear(
        A, B, C, D, skip_controllability_check=True, skip_observability_check=True
    )

    t1, y1 = system_dynamics1.step_response(step_size=0.01, duration=1.0)
    t2, y2 = system_dynamics2.step_response(step_size=0.01, duration=1.0)

    assert torch.allclose(t1, t2)
    assert torch.allclose(y1, y2)
