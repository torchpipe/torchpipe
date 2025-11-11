import pytest
import omniback._C as _C
from omniback import Dict as omniback_dict

def test_dict_basic():
    # Test construction
    d = _C.Dict()
    assert len(d) == 0
    
    # Test with initial data
    d = _C.Dict({"a": 1, "b": "two"})
    assert d["a"] == 1
    assert d["b"] == "two"

def test_dict_operations():
    d = _C.Dict()
    
    # Test set/get
    d["key"] = "value"
    assert d["key"] == "value"
    
    # Test update
    d.update({"new_key": "new_value"})
    assert d["new_key"] == "new_value"
    
    # Test pop
    value = d.pop("key")
    assert value == "value"
    assert "key" not in d
    
    # Test clear
    d.clear()
    assert len(d) == 0

def test_dict_types():
    d = _C.Dict()
    
    # Test different value types
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
            assert d[key] == value

def test_event():
    # Test event creation
    event = _C.Event()
    d = _C.Dict()
    d["event"] = event
    
if __name__ == "__main__":
    test_event()
