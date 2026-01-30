

from omniback.group_registry import toml2groups
import omniback 
import pytest

def  cb_func(name):
    raise TypeError(name)

def test_register_backend_group():
    toml_path = 'config/group-torchpipe.toml'
    grps, _ = toml2groups(toml_path)
    _register_backend_group = omniback.ffi.register_backend_group
    for backend, grp_names in grps.items():
        assert len(
            grp_names) == 1, f"backend {backend} has multiple groups: {grp_names}"
        grp_name = next(iter(grp_names))
        _register_backend_group(backend, grp_name,
                                lambda: cb_func(backend))
        
        with pytest.raises(TypeError) as exc_info:
            omniback.init(backend)

        assert str(exc_info.value) == backend

if __name__ == "__main__":
    test_register_backend_group()