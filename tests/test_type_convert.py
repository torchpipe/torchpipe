import pytest
import omniback


def test_type_convert_int_to_float():
    d = omniback.Dict()
    d["int_value"] = 42
    d["float_value"] = float(d["int_value"])
    assert d["float_value"] == 42.0


def test_type_convert_float_to_int():
    d = omniback.Dict()
    d["float_value"] = 3.14
    d["int_value"] = int(d["float_value"])
    assert d["int_value"] == 3


def test_type_convert_string_to_int():
    d = omniback.Dict()
    d["string_value"] = "123"
    d["int_value"] = int(d["string_value"])
    assert d["int_value"] == 123


def test_type_convert_string_to_float():
    d = omniback.Dict()
    d["string_value"] = "3.14"
    d["float_value"] = float(d["string_value"])
    assert pytest.approx(d["float_value"]) == 3.14


def test_type_convert_bool_to_int():
    d = omniback.Dict()
    d["bool_value"] = True
    d["int_value"] = int(d["bool_value"])
    assert d["int_value"] == 1


def test_type_convert_int_to_bool():
    d = omniback.Dict()
    d["int_value"] = 1
    d["bool_value"] = bool(d["int_value"])
    assert d["bool_value"] == True


def test_type_bytes_to_string():
    d = omniback.Dict()
    d["bytes_value"] = b"hello"
    d["string_value"] = d["bytes_value"].decode("utf-8")
    assert d["string_value"] == "hello"


def test_type_string_to_bytes():
    d = omniback.Dict()
    d["string_value"] = "hello"
    d["bytes_value"] = d["string_value"].encode("utf-8")
    assert d["bytes_value"] == b"hello"


if __name__ == "__main__":
    test_type_convert_int_to_float()
    test_type_convert_float_to_int()
    test_type_convert_string_to_int()
    test_type_convert_string_to_float()
    test_type_convert_bool_to_int()
    test_type_convert_int_to_bool()
    test_type_bytes_to_string()
    test_type_string_to_bytes()
    print("All type convert tests passed!")
