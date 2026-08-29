import pytest

from opencodelab.cli import _validate_gpu


def test_single_gpu_selectors_are_allowed():
    assert _validate_gpu("L40S") == "L40S"
    assert _validate_gpu("H100") == "H100"


def test_multi_gpu_selector_is_rejected_until_tensor_parallel_is_supported():
    with pytest.raises(ValueError):
        _validate_gpu("H100:2")
