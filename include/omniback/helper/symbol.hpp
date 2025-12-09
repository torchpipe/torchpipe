#pragma once

#include <string>
#include "omniback/helper/omniback_export.h"
namespace omniback {
std::string local_demangle(const char* name);

template <class T>
std::string type(const T& t) {
  return local_demangle(typeid(t).name());
}

OMNI_EXPORT void throw_wrong_type(
    const char* need_type,
    const char* input_type);
} // namespace omniback