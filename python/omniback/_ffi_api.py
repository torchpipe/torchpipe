
import tvm_ffi


# this is a short cut to register all the global functions
tvm_ffi.init_ffi_api("omniback", __name__)
