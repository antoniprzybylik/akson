import pytest
import torch

from akson._validation import (
    validate_tensor_shape_with_names,
    validate_tensor,
    validate_and_move_optional_tensor,
    validate_optional_tensors_le,
)


def test_validate_tensor_shape_with_names_passes_on_correct_shape():
    t = torch.zeros(3, 4)
    validate_tensor_shape_with_names(
        t, "t", (3, 4), ("rows", "cols"), ("number of rows", "number of cols")
    )


def test_validate_tensor_shape_with_names_passes_with_wildcard_dims():
    t = torch.zeros(7, 4)  # first dimension is unconstrained (None)
    validate_tensor_shape_with_names(
        t, "t", (None, 4), ("n", "cols"), ("anything", "number of cols")
    )


def test_validate_tensor_shape_with_names_raises_on_wrong_ndim():
    t = torch.zeros(3, 4, 5)
    with pytest.raises(ValueError):
        validate_tensor_shape_with_names(
            t, "t", (3, 4), ("rows", "cols"), ("number of rows", "number of cols")
        )


def test_validate_tensor_shape_with_names_raises_on_wrong_fixed_dim():
    t = torch.zeros(3, 5)
    with pytest.raises(ValueError):
        validate_tensor_shape_with_names(
            t, "t", (3, 4), ("rows", "cols"), ("number of rows", "number of cols")
        )


def test_validate_tensor_shape_with_names_error_message_contains_tensor_name():
    t = torch.zeros(3, 5)
    with pytest.raises(ValueError, match="t must have shape"):
        validate_tensor_shape_with_names(
            t, "t", (3, 4), ("rows", "cols"), ("number of rows", "number of cols")
        )


def test_validate_tensor_shape_with_names_error_message_contains_actual_shape():
    t = torch.zeros(3, 5)
    with pytest.raises(ValueError, match=r"Got shape torch.Size\(\[3, 5\]\)"):
        validate_tensor_shape_with_names(
            t, "t", (3, 4), ("rows", "cols"), ("number of rows", "number of cols")
        )


def test_validate_tensor_passes_on_correct_shape():
    t = torch.zeros(2, 3)
    validate_tensor(t, "t", (2, 3))


def test_validate_tensor_raises_on_wrong_shape():
    t = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="Expected t to have shape"):
        validate_tensor(t, "t", (3, 2))


def test_validate_tensor_raises_on_wrong_ndim():
    t = torch.zeros(2, 3)
    with pytest.raises(ValueError):
        validate_tensor(t, "t", (2, 3, 1))


def test_validate_and_move_optional_tensor_returns_none_for_none_input():
    result = validate_and_move_optional_tensor(
        None, "x", (3,), desired_dtype=torch.float64, desired_device=torch.device("cpu")
    )
    assert result is None


def test_validate_and_move_optional_tensor_passes_correct_shape():
    t = torch.zeros(3, dtype=torch.float32)
    result = validate_and_move_optional_tensor(
        t, "x", (3,), desired_dtype=torch.float64, desired_device=torch.device("cpu")
    )
    assert result.shape == (3,)


def test_validate_and_move_optional_tensor_casts_dtype():
    t = torch.zeros(3, dtype=torch.float32)
    result = validate_and_move_optional_tensor(
        t, "x", (3,), desired_dtype=torch.float64, desired_device=torch.device("cpu")
    )
    assert result.dtype == torch.float64


def test_validate_and_move_optional_tensor_raises_on_wrong_shape():
    t = torch.zeros(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="x has bad shape"):
        validate_and_move_optional_tensor(
            t,
            "x",
            (3,),
            desired_dtype=torch.float64,
            desired_device=torch.device("cpu"),
        )


def test_validate_and_move_optional_tensor_error_message_contains_name():
    t = torch.zeros(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="x has bad shape"):
        validate_and_move_optional_tensor(
            t,
            "x",
            (3,),
            desired_dtype=torch.float64,
            desired_device=torch.device("cpu"),
        )


def test_validate_optional_tensors_le_passes_when_both_none():
    validate_optional_tensors_le(None, "a", None, "b")


def test_validate_optional_tensors_le_passes_when_first_none():
    t = torch.tensor([1.0, 2.0])
    validate_optional_tensors_le(None, "a", t, "b")


def test_validate_optional_tensors_le_passes_when_second_none():
    t = torch.tensor([1.0, 2.0])
    validate_optional_tensors_le(t, "a", None, "b")


def test_validate_optional_tensors_le_passes_when_elementwise_le():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([1.0, 3.0])
    validate_optional_tensors_le(a, "a", b, "b")


def test_validate_optional_tensors_le_raises_when_violated():
    a = torch.tensor([1.0, 5.0])
    b = torch.tensor([1.0, 3.0])
    with pytest.raises(ValueError, match="a must be <= b elementwise"):
        validate_optional_tensors_le(a, "a", b, "b")


def test_validate_optional_tensors_le_equal_values_pass():
    a = torch.tensor([2.0, 2.0])
    b = torch.tensor([2.0, 2.0])
    validate_optional_tensors_le(a, "a", b, "b")
