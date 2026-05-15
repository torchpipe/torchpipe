import pytest
import omniback
import tempfile
import os


def test_parser_pipe_with_simple_string():
    config_str = "Identity"
    model = omniback.pipe(config_str)
    assert model is not None


def test_parser_pipe_with_single_node():
    config = {
        'processor': {}
    }
    model = omniback.pipe(config)
    assert model is not None


def test_parser_pipe_with_chain():
    config = {
        'preprocess': {'next': "inference"},
        'inference': {}
    }
    model = omniback.pipe(config)
    assert model is not None


def test_parser_init_from_file():
    config_content = "[jpg_decoder]\nbackend = \"Identity\"\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(config_content)
        temp_path = f.name

    try:
        model = omniback.init_from_file(temp_path)
        assert model is not None
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    test_parser_pipe_with_simple_string()
    test_parser_pipe_with_single_node()
    test_parser_pipe_with_chain()
    test_parser_init_from_file()
    print("All parser tests passed!")
