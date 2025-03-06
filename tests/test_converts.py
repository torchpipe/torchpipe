import pytest
import hami._C as _C

def test_type_converts():
    # Test basic type conversions
    test_cases = [
        (42, int),
        (3.14, float),
        ("test", str),
        (True, int),
        ([1, 2, 3], list),
        ({"a": 1}, dict)
    ]
    
    for value, expected_type in test_cases:
        any_obj = _C.Any(value)
        assert isinstance(any_obj.value, expected_type)

def test_numeric_converts():
    # Test numeric conversions
    num = _C.Any(42)
    
    # Integer conversions
    assert isinstance(num.value, int)
    
    # Float conversions
    float_num = _C.Any(3.14)
    assert isinstance(float_num.value, float)
    assert float_num.value == pytest.approx(3.14)

def test_container_converts():
    # Test list conversions
    list_data = [1, "two", 3.0]
    with pytest.raises(RuntimeError):
        list_any = _C.Any(list_data)
    
    dict_data = {"a": 1.0, "b": 2.0, "c": 3.0}
    dict_any = _C.Any(dict_data)
    for key, value in dict_data.items():
        if isinstance(value, float):
            assert dict_any.value[key] == pytest.approx(value)
        else:
            assert dict_any.value[key] == value