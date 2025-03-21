import pytest
import hami._C as _C

def test_backend_creation():
    # Test Identity backend
    backend = _C.create("Identity")
    assert backend is not None
    assert backend.max() == 18446744073709551615

def test_backend_initialization():
    backend = _C.create("Identity")
    # Test chained initialization
    backend.init({"param1": "value1"}).init({})
    
    # Test with empty config
    backend.init({})

def test_backend_execution():
    backend = _C.create("Identity")
    backend.init({})
    
    # Test with different input types
    test_cases = [
        {"data": "string_input"},
        {"data": 42},
        {"data": 3.14},
        {"data": [1, 2, 3]},
        {"data": {"nested": "data"}}
    ]
    
    for test_input in test_cases:
        input_dict = _C.Dict(test_input)
        backend(input_dict)
        assert pytest.approx(input_dict["result"]) == test_input["data"]

def test_backend_with_event():
    backend = _C.create("Identity")
    backend.init({})
    
    # Test with event
    input_data = _C.Dict({"data": 100, "event": 1})
    backend(input_data)
    assert input_data["result"] == 100