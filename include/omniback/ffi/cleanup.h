#pragma once
#include <functional>
#include "omniback/helper/omniback_export.h"

namespace om::ffi {
    void OMNI_EXPORT atexit_register(std::function<void()> clear_func);
}