import pytest
import omniback as om

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
        any_obj = om.Dict({'1':value})
        # print(any_obj['1'], type(any_obj['1']))
        assert (expected_type(any_obj['1']) == value)
        # assert isinstance(any_obj['1'], expected_type)



if __name__ == "__main__":
    test_type_converts()
