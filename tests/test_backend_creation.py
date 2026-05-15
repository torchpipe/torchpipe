import pytest
import omniback


def test_backend_create_identity():
    backend = omniback.create("Identity")
    assert backend is not None
    backend.init({}, None)


def test_backend_forward_with_string():
    backend = omniback.create("Identity")
    backend.init({}, None)
    io = {"data": "test_string"}
    backend(io)
    assert io["result"] == "test_string"


def test_backend_forward_with_int():
    backend = omniback.create("Identity")
    backend.init({}, None)
    io = {"data": 12345}
    backend(io)
    assert io["result"] == 12345


def test_backend_forward_with_float():
    backend = omniback.create("Identity")
    backend.init({}, None)
    io = {"data": 3.14159}
    backend(io)
    assert pytest.approx(io["result"]) == 3.14159


def test_backend_forward_multiple_times():
    backend = omniback.create("Identity")
    backend.init({}, None)

    for i in range(10):
        io = {"data": i}
        backend(io)
        assert io["result"] == i


if __name__ == "__main__":
    test_backend_create_identity()
    test_backend_forward_with_string()
    test_backend_forward_with_int()
    test_backend_forward_with_float()
    test_backend_forward_multiple_times()
    print("All backend tests passed!")
