#pragma once

#include <string>
#include "hami/helper/hami_export.h"
namespace hami {
std::string local_demangle(const char* name);

template <class T>
std::string type(const T& t) {
    return local_demangle(typeid(t).name());
}

HAMI_EXPORT void throw_wrong_type(const char* need_type,
                                  const char* input_type);
}  // namespace hami