import pytest
import omniback as om
import omniback
import numpy as np

def test_backend():
    bk = om.Backend()
    
def test_backend_creation():
    # Test Identity backend
    backend = omniback.create("Identities", None)
    assert backend is not None
    print(backend.max())
    assert backend.max() == np.iinfo(np.uint32).max
    backend.init({'max': "5"}, {})
    assert backend.max() == 5

def test_backend_initialization():
    backend = om.create("Identity")
    # Test chained initialization
    a=backend.init({"param1": "value1"}, None)
    assert isinstance(a, om.Backend)
    
    # Test with empty config
    backend.init({}, None)
    


def test_backend_execution():
    backend = om.create("Identity")
    backend.init({}, None)
    
    # Test with different input types
    test_cases = [
        {"data": "string_input"},
        {"data": 42},
        {"data": 3.14},
        {"data": [1, 2, 3]},
        {"data": {"nested": "data"}}
    ]
    
    for test_input in test_cases:
        input_dict = test_input # om.Dict(test_input)
        backend(input_dict)
        assert pytest.approx(input_dict["result"]) == test_input["data"]

def test_backend_with_event():
    backend = om.create("Identity")
    backend.init({}, None)
    
    input_data = om.Dict({"data": 100, "event": 1})
    backend(input_data)
    assert input_data["result"] == 100

    with pytest.raises(RuntimeError):
        io = {"data": 100, "event": om.Event()}
        backend(io)

        
    
if __name__ == "__main__":
    test_backend_with_event()
