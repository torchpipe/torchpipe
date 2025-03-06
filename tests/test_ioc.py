import pytest
import hami._C as _C
import hami
def test_ioc_initialization():
    # Test IoC container initialization
    ioc = _C.create("IoC")
    assert ioc is not None

    # Test initialization with valid configuration
    config = {
        "IoC::dependency": "Identity;Identity"
    }
    dict_config = {}
    ioc.init(config)

    # Test initialization with missing dependency configuration
    with pytest.raises(Exception):
        invalid_config = {"param1": "value1"}
        ioc.init(invalid_config, dict_config)

def test_ioc_forward():
    # Test IoC container forward method
    ioc = _C.create("IoC")
    config = {
        "IoC::dependency": "Identity;Identity",
        "Identity::param1": "value1"
    }
    dict_config = {}
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
        input_dict = _C.Dict(test_input)
        ioc(input_dict)
        assert pytest.approx(input_dict["result"]) == test_input["data"]


def test_ioc_phase_initialization():
    # Test IoC container phase initialization
    ioc = _C.create("IoC")
    config = {
        "IoC::dependency": "Identity,Identity;Identity",
        "BackendA::param1": "value1",
        "BackendB::param2": "value2",
        "BackendC::param3": "value3"
    }
    dict_config = {}
    # ioc.init(config, hami.Dict())
    with pytest.raises(RuntimeError):
        ioc.init(config, hami.Dict())

 
    
    
if __name__ == "__main__":
    test_ioc_phase_initialization()