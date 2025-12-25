import pytest
# from omniback import _C


import omniback
import omniback as om

def test_ioc_initialization():
    # Test IoCV0 container initialization
    ioc = om.create("IoCV0")
    assert ioc is not None

    # Test initialization with valid configuration
    config = {
        "IoCV0::dependency": "Identity;Forward[Identity]"
    }
    kwargs = {}
    ioc.init(config)

    # Test initialization with missing dependency configuration
    with pytest.raises(Exception):
        invalid_config = {"param1": "value1"}
        ioc.init(invalid_config, kwargs)

def test_ioc_forward():
    # Test IoCV0 container forward method
    ioc = om.create("IoCV0")
    config = {
        "IoCV0::dependency": "Identity;Forward[Identity]",
        "Identity::param1": "value1"
    }
    kwargs = {}
    ioc.init(config)

    # Test forward with different input types
    test_cases = [
        {"data": "string_input"},
        {"data": 42},
        {"data": 3.14},
        {"data": [1, 2, 3]},
        {"data": {"nested": "data"}}
    ]

    for test_input in test_cases:
        input_dict = om.Dict(test_input)
        ioc(input_dict)
        assert pytest.approx(input_dict["result"]) == test_input["data"]


def test_ioc_phase_initialization():
    # Test IoCV0 container phase initialization
    ioc = om.create("IoCV0")
    config = {
        "IoCV0::dependency": "Identity,Identity;Forward[Identity]",
        "BackendA::param1": "value1",
        "BackendB::param2": "value2",
        "BackendC::param3": "value3"
    }
    kwargs = {}
    # ioc.init(config, omniback.Dict())
    with pytest.raises(RuntimeError):
        ioc.init(config, omniback.Dict())
    # ioc.init(config, omniback.Dict())

 
    
    
if __name__ == "__main__":
    test_ioc_phase_initialization()