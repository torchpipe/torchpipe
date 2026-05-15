"""Tests for omniback Dict and Event types."""
import pytest
from omniback import Dict
import omniback as om
import tvm_ffi


def test_dict_basic():
    """Test basic Dict construction."""
    d = Dict()
    assert len(d) == 0

    d = Dict({"a": 1, "b": "two"})
    assert d["a"] == 1
    assert d["b"] == "two"


def test_dict_operations():
    """Test Dict CRUD operations."""
    d = Dict()

    d["key"] = "value"
    assert d["key"] == "value"

    d.update({"new_key": "new_value"})
    assert d["new_key"] == "new_value"

    e = Dict({"another_key": 123})
    d.update(e)
    assert d["another_key"] == 123

    value = d.pop("key")
    assert value == "value"
    e = d.copy()
    assert isinstance(e, type(d))
    assert "key" not in e
    assert "another_key" in e

    d.clear()
    assert len(d) == 0


def test_dict_types():
    """Test Dict with various value types."""
    d = Dict()

    test_values = {
        "int": 42,
        "float": 3.14,
        "str": "test",
        "bool": True,
        "list": [1, 2, 3],
        "dict": {"nested": "value"}
    }

    for key, value in test_values.items():
        d[key] = value
        if isinstance(value, float):
            assert d[key] == pytest.approx(value)
        else:
            assert str(d[key]) == str(tvm_ffi.convert(value))


def test_event():
    """Test Event creation and usage."""
    d = Dict()
    event = om.Event()

    d["event"] = event
    d["event"].wait(10)
    assert isinstance(d["event"], om.Event)


def test_any():
    """Test Any type flexibility in Dict."""
    data = om.Dict({"1": 1})
    assert data["1"] == 1 and isinstance(data["1"], int)

    data["1"] = "33"
    assert data["1"] == "33" and isinstance(data["1"], str)

    data["2"] = "33"
    assert data["2"] == "33" and isinstance(data["2"], str) and data['1'] == '33'
    assert len(data) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
