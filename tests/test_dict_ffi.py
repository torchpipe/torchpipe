import pytest
import omniback


def test_dict_basic():
    d = omniback.Dict()
    d["key1"] = "value1"
    d["key2"] = 42
    assert d["key1"] == "value1"
    assert d["key2"] == 42


def test_dict_nested():
    d = omniback.Dict()
    d["nested"] = {"inner_key": "inner_value"}
    assert d["nested"]["inner_key"] == "inner_value"


def test_dict_contains():
    d = omniback.Dict()
    d["key1"] = "value1"

    assert "key1" in d
    assert "key2" not in d


def test_dict_delete():
    d = omniback.Dict()
    d["key1"] = "value1"
    del d["key1"]

    assert "key1" not in d


def test_dict_iteration():
    d = omniback.Dict()
    d["key1"] = "value1"
    d["key2"] = "value2"

    keys = list(d.keys())
    assert "key1" in keys
    assert "key2" in keys


if __name__ == "__main__":
    test_dict_basic()
    test_dict_nested()
    test_dict_contains()
    test_dict_delete()
    test_dict_iteration()
    print("All dict tests passed!")
