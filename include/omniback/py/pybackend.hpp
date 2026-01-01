#pragma once

#include <tvm/ffi/function.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/container/variant.h>
#include "omniback/ffi/dict.h"

namespace omniback::py {

using SelfType = tvm::ffi::Any; // tvm::ffi::ObjectPtr<tvm::ffi::Object>;
using PyDictRef = omniback::ffi::DictRef;

std::unique_ptr<omniback::Backend> object2backend(
    SelfType py_obj,
    tvm::ffi::Optional<tvm::ffi::TypedFunction<void(
        SelfType,
        const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::String>&,
        tvm::ffi::Optional<PyDictRef>)>> init_func,
    tvm::ffi::Optional<
        tvm::ffi::TypedFunction<void(SelfType, tvm::ffi::Array<PyDictRef>)>>
        forward_func,
    tvm::ffi::Optional<tvm::ffi::Variant<
        tvm::ffi::TypedFunction<uint32_t(SelfType)>,
        uint32_t>> max_func,
    tvm::ffi::Optional<tvm::ffi::Variant<
        tvm::ffi::TypedFunction<uint32_t(SelfType)>,
        uint32_t>> min_func);
}