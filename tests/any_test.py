import pytest
import hami._C as _C
from hami import Dict as hami_dict

def test_any_basic_types():
    # Integer
    a = _C.Any(1)
    assert a.value == 1
    
    # Large integer overflow
    with pytest.raises(Exception):
        _C.Any(321112222222321112222222)
    
    # Double
    try:
        f = _C.Any(32323423423423423.3)
    except:
        f = _C.Any()
        f.set_double(3232342342342342.1)
        val = f.as_double()
        assert isinstance(val, float)

def test_any_containers():
    # String list
    str_list = _C.Any(["dd"])
    assert str_list.value[0] == "dd"
    
    # Integer list
    int_list = _C.Any([1, 2])
    assert isinstance(int_list.value, list)
    assert int_list.value == [1, 2]

    # Dictionary with float values
    test_dict = {"dd": 21.1, "ddd": 2.2}
    dict_any = _C.Any(test_dict)
    result_dict = dict_any.value
    
    assert len(result_dict) == len(test_dict)
    for key, value in test_dict.items():
        if isinstance(value, float):
            assert abs(result_dict[key] - value) < 1e-6
        else:
            assert result_dict[key] == value

def test_dict_operations():
    d = _C.Dict({"key1": "value1"})
    d['key2'] = 2
    d['key3'] = "value3"
    
    assert d['key2'] == 2
    assert d['key3'] == "value3"
    
    d.pop('key2')
    assert 'key2' not in d
    assert len(d) == 2

def test_identity_backend():
    backend = _C.create("Identity")
    backend.init({"param1": "value1"}).init({})
    
    # String input
    input_str = _C.Dict({"data": "test_string"})
    backend(input_str)
    assert input_str['result'] == "test_string"
    assert backend.max() == 18446744073709551615
    
    # Integer input
    input_int = {"data": 42}
    backend(input_int)
    
    # Input with event
    input_event = {"data": 100, 'event': 1}
    input_event = hami_dict(input_event)
    backend(input_event)
    assert input_event['result'] == 100

if __name__ == "__main__":
    pytest.main([__file__])