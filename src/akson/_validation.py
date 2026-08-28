from typing import List, Optional
import torch


def validate_tensor_shape_with_names(
    tensor: torch.Tensor,
    tensor_name: str,
    sizes: List[Optional[int]],
    size_names: List[str],
    size_descs: List[str],
) -> None:
    ok = True
    if tensor.ndim != len(sizes):
        ok = False
    else:
        for i, size in enumerate(sizes):
            if size is not None and tensor.shape[i] != size:
                ok = False
    if not ok:
        size_names_and_values = [
            str(a) if a is not None else b for (a, b) in zip(sizes, size_names)
        ]
        msg = (
            f"{tensor_name} must have shape ({', '.join(size_names_and_values)}) where "
        )
        comma_needed = False
        for size, size_name, size_desc in zip(sizes, size_names, size_descs):
            if size is None:
                if comma_needed:
                    msg += ", "
                msg += f"`{size_name}` is the {size_desc}"
                comma_needed = True
        msg += f". Got shape {tensor.shape}"
        raise ValueError(msg)


def validate_tensor(tensor: torch.Tensor, name: str, expected_shape: List[int]) -> None:
    if tensor.shape != expected_shape:
        raise ValueError(
            f"Expected {name} to have shape {expected_shape}. Got {tensor.shape}"
        )


def validate_and_move_optional_tensor(
    tensor: Optional[torch.Tensor],
    tensor_name: str,
    expected_shape: List[int],
    desired_dtype: torch.dtype,
    desired_device: torch.device,
) -> Optional[torch.Tensor]:
    if tensor is not None:
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{tensor_name} has bad shape {tensor.shape}. Expected {expected_shape}"
            )
        return tensor.to(dtype=desired_dtype, device=desired_device)
    else:
        return None


def validate_optional_tensors_le(
    tensor1: torch.Tensor, name1: str, tensor2: torch.Tensor, name2: str
) -> None:
    if tensor1 is not None and tensor2 is not None and (tensor1 > tensor2).any():
        raise ValueError(f"{name1} must be <= {name2} elementwise, but it is not")
