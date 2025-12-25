import pytest
from omniback import ffi
# from omniback.ffi import Dict as omniback_dict
import omniback as om

        

def test_any_containers():


    # Dictionary with float values
    test_dict = {"dd": 21.1, "ddd": 2.2}
    result_dict = om.Dict(test_dict)
    
    assert len(result_dict) == len(test_dict)
    for key, value in test_dict.items():
        assert result_dict[key] == value

def test_dict_operations():
    d = om.Dict({"key1": "value1"})
    d['key2'] = 2
    d['key3'] = "value3"
    
    assert d['key2'] == 2
    assert d['key3'] == "value3"
    
    d.pop('key2')
    assert 'key2' not in d
    assert len(d) == 2

def test_identity_backend():
    backend = om.create("Identity")
    backend.init({"param1": "value1"}).init({})
    
    # String input
    input_str = om.Dict({"data": "test_string"})
    backend(input_str)
    assert input_str['result'] == "test_string"
    assert backend.max() == 1
    
    # Integer input
    input_int = {"data": 42}
    backend(input_int)
    
    # Input with event
    input_event = {"data": 100, 'event': 1}
    input_event = om.Dict(input_event)
    backend(input_event)
    assert input_event['result'] == 100

if __name__ == "__main__":
    test_any_containers()
