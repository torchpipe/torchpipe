from omniback.group_registry import toml2groups
import tvm_ffi


from ._build_trt import _load_or_build_lib

# def setting_groups(toml_path: str):
#     _backend_to_groups, _ = toml2groups(toml_path)
#     for backend, grp_name in _backend_to_groups.items():
#             assert len(grp_name) == 1, f"backend {backend} has multiple groups: {grp_name}"
#             _register_backend_group = tvm_ffi.get_global_func(
#                 "omniback.register_backend_group")
#             _register_backend_group(backend, grp_name.first(),
#                                 lambda: cb(backend, grp_name.first()))