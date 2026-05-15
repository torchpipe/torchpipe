import pytest
import omniback as om

def test_backend_input_variations():
    """Test backend execution with various input types."""
    # Use Identities for batch support
    backend = om.create("Identities")
    backend.init({"max": "10"}, None)

    # 1. Single Dict
    d1 = om.Dict({"data": 1})
    backend(d1)
    assert d1["result"] == 1

    # 2. List of Dicts
    d2 = om.Dict({"data": 2})
    d3 = om.Dict({"data": 3})
    backend([d2, d3])
    assert d2["result"] == 2
    assert d3["result"] == 3

    # 3. Invalid input type
    with pytest.raises(TypeError):
        backend("invalid")

    # 4. Python Dict input
    # Supported via auto-conversion, updates are reflected in-place
    d4 = {"data": 4}
    backend(d4)
    assert d4["result"] == 4

if __name__ == "__main__":
    test_backend_input_variations()
