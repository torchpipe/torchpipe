def test_tf():
    import tvm_ffi
    mod = tvm_ffi.cpp.load_inline(name='hello',
                                cpp_sources="""
                            #include <limits>
                            tvm::ffi::Any example_func() {
                                return std::numeric_limits<uint32_t>::max();
                            }
                            """,
                                functions=['example_func'])

    result = mod.example_func()

    print(f'result={result} type={type(result)}')


def test_repr():
    return
    import tvm_ffi
    # @tvm_ffi.register_object("omniback.SharedAny")
    # class SharedAny(tvm_ffi.Object):
    #     def __init__(self) -> None:
    #         """Construct the object."""
    #         self.__ffi_init__()
    #     def __repr__(self):
    #         return 
    
    mod = tvm_ffi.cpp.load_inline(name='hello',
                                  cpp_sources="""
    #include <tvm/ffi/container/map.h>
    #include <tvm/ffi/reflection/registry.h> 
    struct AnyObj : public tvm::ffi::Object {
    AnyObj(){}
    AnyObj(tvm::ffi::Array<tvm::ffi::Any>&& data) : data(std::move(data)){}
    tvm::ffi::Array<tvm::ffi::Any> data;
    static constexpr bool _type_mutable = true;
    TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
        /*type_key=*/"test.SharedAny",
        /*class=*/AnyObj,
        /*parent_class=*/tvm::ffi::Object);
    };

    TVM_FFI_STATIC_INIT_BLOCK() {
        namespace refl = tvm::ffi::reflection;
        refl::ObjectDef<AnyObj>();
    }

        tvm::ffi::Any example_func(tvm::ffi::Any data) {
            auto array_obj1 = tvm::ffi::Array<tvm::ffi::Any>();
            
            auto array_obj2 = tvm::ffi::Array<tvm::ffi::Any>();
            array_obj1.push_back(data);
            array_obj2.push_back(array_obj1);
            return data.get(array_obj1);
        }
        """,
                functions=['example_func'])

    a = dict()
    result = mod.example_func(a)
    a[1] = result
    print(result)
    
if __name__ == "__main__":
    test_repr()
