"""Basic import and version tests for torchpipe."""
import omniback
import torchpipe


def test_import():
    """Test that torchpipe can be imported."""
    assert torchpipe is not None


def test_version():
    """Test that version is accessible."""
    assert hasattr(torchpipe, "__version__")
    assert isinstance(torchpipe.__version__, str)


def test_reexport():
    """Test that torchpipe re-exports omniback APIs."""
    assert hasattr(torchpipe, "pipe")
    assert hasattr(torchpipe, "Dict")
    assert hasattr(torchpipe, "register")


if __name__ == "__main__":
    test_import()
    test_version()
    test_reexport()
    print("All tests passed!")
