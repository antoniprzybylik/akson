import torch
from typing import Optional, Tuple

def build_dynamic_matrices(
    step_response: torch.Tensor,
    N: int,
    Nu: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """! Builds dynamic matrix M and matrix of past dynamics Mp for DMC and QDMC regulators.

    @param step_response Discrete step response
        Shape: (D, n_outputs, n_inputs)
    @param N Prediction horizon
    @param Nu Control horizon

    @return M Dynamic matrix
        Shape: (N*n_outputs, Nu*n_inputs)
    @return Mp Matrix of past dynamics
        Shape: (N*n_outputs, (D-1)*n_inputs)
    """
    S = step_response
    D = S.shape[0]
    n_outputs = S.shape[1]
    n_inputs = S.shape[2]
    dtype = S.dtype
    device = S.device

    M = torch.zeros(N * n_outputs, Nu * n_inputs, dtype=dtype, device=device)
    for i in range(1, N + 1):
        for j in range(1, Nu + 1):
            k = i - j + 1
            if k >= 0:
                rows_slice = slice((i - 1) * n_outputs, i * n_outputs)
                columns_slice = slice((j - 1) * n_inputs, j * n_inputs)
                M[rows_slice, columns_slice] = S[min(k, D - 1)]

    Mp = torch.zeros(N * n_outputs, (D - 1) * n_inputs, dtype=dtype, device=device)
    for i in range(1, D):
        for j in range(1, N + 1):
            rows_slice = slice((j - 1) * n_outputs, j * n_outputs)
            columns_slice = slice((i - 1) * n_inputs, i * n_inputs)
            Mp[rows_slice, columns_slice] = S[min(i + j, D - 1)] - S[i]

    return M, Mp


def build_sum_input_deltas_array(
    N: int,
    Nu: int,
    n_inputs: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """! Constructs matrix that sums input signal deltas
    The sum_input_deltas_array is a matrix that can be used to create
    the prefix sums of control increments (du).

    @return sum_input_deltas_array
        Shape: (N*n_inputs, Nu*n_inputs)
    """
    I = torch.eye(n_inputs, dtype=dtype, device=device)
    Z = torch.zeros((n_inputs, n_inputs), dtype=dtype, device=device)
    rows = []
    for i in range(N):
        row = torch.cat([I if i >= j else Z for j in range(Nu)], dim=1)
        rows.append(row)
    return torch.cat(rows, dim=0)


def zero_past_du_and_current_u(
    dynamics_horizon: int,
    n_inputs: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    past_du = torch.zeros(dynamics_horizon - 1, n_inputs, dtype=dtype, device=device)
    current_u = torch.zeros(n_inputs, dtype=dtype, device=device)
    return past_du, current_u


class BaseDMCState:
    """! Base class for DMC family regulator states. """
    def __init__(
        self,
        past_du: torch.Tensor,
        current_u: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        if past_du.ndim != 2:
            raise ValueError(f"past_du has bad shape {past_du.shape}. Expected two dimensions.")
        if current_u.ndim != 1:
            raise ValueError(f"current_u has bad shape {current_u.shape}. Expected one dimension.")
        if past_du.shape[1] != current_u.shape[0]:
            raise ValueError("current_u and past_du dimensions do not conform.")

        if dtype is not None:
            past_du = past_du.to(dtype)
            current_u = current_u.to(dtype)
        if device is not None:
            past_du = past_du.to(device)
            current_u = current_u.to(device)

        self.past_du = past_du
        self.current_u = current_u
        self.dtype = past_du.dtype
        self.device = past_du.device
