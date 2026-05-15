

from omniback.group_registry import toml2groups
import omniback
import pytest
import os

def cb_func(name):
    raise TypeError(name)

def test_register_backend_group():
    toml_path = os.path.join(os.path.dirname(__file__), 'config/group-torchpipe.toml')
    grps, _ = toml2groups(toml_path)
    _register_backend_group = omniback.ffi.register_backend_group
    for backend, grp_names in grps.items():
        assert len(
            grp_names) == 1, f"backend {backend} has multiple groups: {grp_names}"
        grp_name = next(iter(grp_names))
        _register_backend_group(backend, grp_name,
                                lambda: cb_func(backend))

        with pytest.raises(RuntimeError) as exc_info:
            omniback.init(backend)

        assert backend in str(exc_info.value)

if __name__ == "__main__":
    test_register_backend_group()
