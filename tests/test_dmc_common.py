import pytest
import torch

from akson._dmc_common import (
    build_dynamic_matrices,
    build_sum_input_deltas_array,
    zero_past_du_and_current_u,
    BaseDMCState,
)


def _siso_step_response():
    # D = 3 samples: S[0] = 0, S[1] = 1.0, S[2] = 1.8
    return torch.tensor([[[0.0]], [[1.0]], [[1.8]]], dtype=torch.float64)


def test_build_dynamic_matrices_shapes():
    S = _siso_step_response()
    M, Mp = build_dynamic_matrices(S, N=2, Nu=2)
    assert M.shape == (2, 2)
    assert Mp.shape == (2, 2)


def test_build_dynamic_matrices_M_values_siso():
    S = _siso_step_response()
    M, _ = build_dynamic_matrices(S, N=2, Nu=2)
    expected_M = torch.tensor([[1.0, 0.0], [1.8, 1.0]], dtype=torch.float64)
    assert torch.allclose(M, expected_M)


def test_build_dynamic_matrices_Mp_values_siso():
    S = _siso_step_response()
    _, Mp = build_dynamic_matrices(S, N=2, Nu=2)
    expected_Mp = torch.tensor([[0.8, 0.0], [0.8, 0.0]], dtype=torch.float64)
    assert torch.allclose(Mp, expected_Mp)


def test_build_dynamic_matrices_M_is_lower_triangular_when_N_equals_Nu():
    S = torch.tensor([[[0.0]], [[0.5]], [[0.9]], [[1.2]]], dtype=torch.float64)
    M, _ = build_dynamic_matrices(S, N=3, Nu=3)
    assert torch.allclose(torch.triu(M, diagonal=1), torch.zeros_like(M))


def test_build_dynamic_matrices_mimo_shapes():
    D, n_outputs, n_inputs = 4, 2, 2
    S = torch.rand(D, n_outputs, n_inputs, dtype=torch.float64)
    N, Nu = 3, 2
    M, Mp = build_dynamic_matrices(S, N, Nu)
    assert M.shape == (N * n_outputs, Nu * n_inputs)
    assert Mp.shape == (N * n_outputs, (D - 1) * n_inputs)


def test_build_dynamic_matrices_preserves_dtype_and_device():
    S = _siso_step_response().to(torch.float32)
    M, Mp = build_dynamic_matrices(S, N=2, Nu=2)
    assert M.dtype == torch.float32
    assert Mp.dtype == torch.float32
    assert M.device == S.device
    assert Mp.device == S.device


def test_build_sum_input_deltas_array_siso_shape():
    arr = build_sum_input_deltas_array(
        N=3, Nu=2, n_inputs=1, dtype=torch.float64, device=torch.device("cpu")
    )
    assert arr.shape == (3, 2)


def test_build_sum_input_deltas_array_siso_values():
    arr = build_sum_input_deltas_array(
        N=3, Nu=2, n_inputs=1, dtype=torch.float64, device=torch.device("cpu")
    )
    expected = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    assert torch.allclose(arr, expected)


def test_build_sum_input_deltas_array_mimo_block_identity():
    n_inputs = 2
    arr = build_sum_input_deltas_array(
        N=2, Nu=2, n_inputs=n_inputs, dtype=torch.float64, device=torch.device("cpu")
    )
    assert arr.shape == (4, 4)

    I = torch.eye(n_inputs, dtype=torch.float64)
    Z = torch.zeros((n_inputs, n_inputs), dtype=torch.float64)
    # First block row (i=0) is [I, 0], second block row (i=1) is [I, I]
    assert torch.allclose(arr[0:2, 0:2], I)
    assert torch.allclose(arr[0:2, 2:4], Z)
    assert torch.allclose(arr[2:4, 0:2], I)
    assert torch.allclose(arr[2:4, 2:4], I)


def test_build_sum_input_deltas_array_dtype_device():
    arr = build_sum_input_deltas_array(
        N=2, Nu=2, n_inputs=1, dtype=torch.float32, device=torch.device("cpu")
    )
    assert arr.dtype == torch.float32
    assert arr.device.type == "cpu"


def test_zero_past_du_and_current_u_shapes():
    past_du, current_u = zero_past_du_and_current_u(
        dynamics_horizon=5, n_inputs=3, dtype=torch.float64, device=torch.device("cpu")
    )
    assert past_du.shape == (4, 3)
    assert current_u.shape == (3,)


def test_zero_past_du_and_current_u_are_zero():
    past_du, current_u = zero_past_du_and_current_u(
        dynamics_horizon=3, n_inputs=2, dtype=torch.float64, device=torch.device("cpu")
    )
    assert torch.allclose(past_du, torch.zeros(2, 2, dtype=torch.float64))
    assert torch.allclose(current_u, torch.zeros(2, dtype=torch.float64))


def test_zero_past_du_and_current_u_dtype():
    past_du, current_u = zero_past_du_and_current_u(
        dynamics_horizon=2, n_inputs=1, dtype=torch.float32, device=torch.device("cpu")
    )
    assert past_du.dtype == torch.float32
    assert current_u.dtype == torch.float32


def test_base_dmc_state_valid_construction():
    past_du = torch.zeros(2, 2, dtype=torch.float64)
    current_u = torch.zeros(2, dtype=torch.float64)
    state = BaseDMCState(past_du, current_u)
    assert torch.allclose(state.past_du, past_du)
    assert torch.allclose(state.current_u, current_u)
    assert state.dtype == torch.float64
    assert state.device == torch.device("cpu")


def test_base_dmc_state_rejects_bad_past_du_ndim():
    past_du = torch.zeros(2, dtype=torch.float64)  # should be 2D
    current_u = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(ValueError):
        BaseDMCState(past_du, current_u)


def test_base_dmc_state_rejects_bad_current_u_ndim():
    past_du = torch.zeros(2, 2, dtype=torch.float64)
    current_u = torch.zeros(2, 1, dtype=torch.float64)  # should be 1D
    with pytest.raises(ValueError):
        BaseDMCState(past_du, current_u)


def test_base_dmc_state_rejects_mismatched_dimensions():
    past_du = torch.zeros(2, 3, dtype=torch.float64)
    current_u = torch.zeros(2, dtype=torch.float64)  # does not match n_inputs=3
    with pytest.raises(ValueError):
        BaseDMCState(past_du, current_u)


def test_base_dmc_state_casts_dtype_and_device():
    past_du = torch.zeros(2, 2, dtype=torch.float32)
    current_u = torch.zeros(2, dtype=torch.float32)
    state = BaseDMCState(past_du, current_u, dtype=torch.float64, device=torch.device("cpu"))
    assert state.past_du.dtype == torch.float64
    assert state.current_u.dtype == torch.float64
