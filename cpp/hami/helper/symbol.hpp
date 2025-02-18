#pragma once

#include <string>

namespace hami {
std::string local_demangle(const char* name);

template <class T>
std::string type(const T& t) {
  return local_demangle(typeid(t).name());
}
}