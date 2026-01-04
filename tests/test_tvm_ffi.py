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
