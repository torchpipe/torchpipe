import pytest
# from omniback.ffi import Dict as omniback_dict
# import omniback as _C
# import omniback._C as _C
from omniback import Dict
from omniback import _C
import omniback as om
import tvm_ffi

def test_dict_basic():
    # Test construction
    d = Dict()
    assert len(d) == 0
    
    # Test with initial data
    d = Dict({"a": 1, "b": "two"})
    assert d["a"] == 1
    assert d["b"] == "two"

def test_dict_operations():
    d = Dict()
    
    # Test set/get
    d["key"] = "value"
    assert d["key"] == "value"
    
    # Test update
    d.update({"new_key": "new_value"})
    assert d["new_key"] == "new_value"
    
    # Test update from Dict
    # with pytest.raises(TypeError):
    e = Dict({"another_key": 123})
    d.update(e)
    assert d["another_key"] == 123

    # Test pop
    value = d.pop("key")
    assert value == "value"
    e = d.copy()
    assert type(e) == type(d)
    assert "key" not in e
    assert "another_key" in e
    
    # Test clear
    d.clear()
    assert len(d) == 0

def test_dict_types():
    d = Dict()
    
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
        print(f'Set d["{key}"] = {value} ({type(d[key])})')
        if isinstance(value, float):
            assert d[key] == pytest.approx(value)
        # elif isinstance(value, list):
        #     print(type(list(d[key])), type(value))
        #     assert list(d[key]) == value
        else:
            assert str(d[key]) == str(tvm_ffi.convert(value)), f" {d[key]} != {tvm_ffi.convert(value)}. {type(d[key])} != {type(tvm_ffi.convert(value))}"
            # assert d[key] == tvm_ffi.convert(
            #     value), f" {d[key]} != {tvm_ffi.convert(value)}. {type(d[key])} != {type(tvm_ffi.convert(value))}"


# def test_event():
#     # Test event creation
#     d = Dict()
#     event = 1

#     d["event"] = 1
#     print(d["event"])
def test_event():
    # Test event creation
    d = Dict()
    event = om.Event()
    
    d["event"] = event
    print(d["event"], type(d["event"]))
    d["event"].wait(10)
    assert isinstance(d["event"], om.Event)

def test_any():
    data = om.Dict({"1": 1})
    assert (data["1"] == 1 and isinstance(data["1"], int))
    data["1"] = "33"
    assert (data["1"] == "33" and isinstance(data["1"], str))
    data["2"] = "33"
    assert (data["2"] == "33" and isinstance(data["2"], str) and data['1'] == '33')
    assert (len(data) == 2)
    print(type(data), data, data["1"])
if __name__ == "__main__":
    import tvm_ffi
    import omniback as om
    import omniback
    
    # omniback.example
    # # tvm_ffi.init_ffi_api("omniback.omniback", "")

    # tvm_ffi.load_module(om.libinfo.find_libomniback())
    example = tvm_ffi.get_global_func("omniback.example")
    # print(example)
    
    test_event()
    # omniback.example()
    # print(om.libinfo.find_libomniback())
    
    
    # om._ffi_api.ffi.example()
    re = example()
    print('re=', re, type(re), re & 0xFFFFFFFFFFFFFFFF)
    # test_any()
