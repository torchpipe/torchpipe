// Copyright 2021-2023 NetEase.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "any2object.hpp"
#include "mat2numpy.hpp"
#include "ipipe_common.hpp"
#include "ipipe_utils.hpp"
#include "dict.hpp"
#include <torch/extension.h>
#include "base_logging.hpp"
#include <ATen/ATen.h>
#include <c10/util/Type.h>
#include <torch/csrc/autograd/python_variable.h>
#include "event.hpp"
#ifdef WITH_OPENCV
#include "opencv2/core.hpp"
#endif

namespace py = pybind11;

namespace ipipe {

#ifdef WITH_OPENCV
bool cast_mat(const TypeInfo& info, const std::string& key, py::dict& result, const any& data) {
  if (typeid(cv::Mat) == info) {
    const cv::Mat* m = any_cast<cv::Mat>(&data);
    if (!m) return false;
    result[py::str(key)] = mat2numpy(*m);

    return true;
  }

  return false;
}
#endif

template <typename T>
struct is_string {
  static const bool value = false;
};
template <>
struct is_string<std::string> {
  static const bool value = true;
};

template <class T, typename std::enable_if<(!(is_string<T>::value)), int>::type = 0>
py::object py_cast(const any& data) {
  const T* result = any_cast<T>(&data);
  if (result) {
    return py::cast(*result);
  }
  IPIPE_THROW(std::string("can not cast from '") + local_demangle(data.type().name()) + " to " +
              local_demangle(typeid(T).name()) + ".");
}
template <class T, typename std::enable_if<((is_string<T>::value)), int>::type = 0>
py::object py_cast(const any& data) {
  const T* result = any_cast<T>(&data);
  if (result) {
    return py::bytes(*result);
  }
  IPIPE_THROW(std::string("can not cast from '") + local_demangle(data.type().name()) + " to " +
              local_demangle(typeid(T).name()) + ".");
}

py::object cast_arithmetic(const any& data) {
  const auto& type = data.type().type();
  if (type == typeid(bool)) {
    return py_cast<bool>(data);
  } else if (type == typeid(float)) {
    return py_cast<float>(data);
  } else if (type == typeid(double)) {
    return py_cast<double>(data);
  } else if (type == typeid(int)) {
    return py_cast<int>(data);
  } else if (type == typeid(unsigned int)) {
    return py_cast<unsigned int>(data);
  } else if (type == typeid(char)) {
    return py_cast<char>(data);
  } else if (type == typeid(unsigned char)) {
    return py_cast<unsigned char>(data);
  }
  IPIPE_THROW(std::string("can not treat '") + local_demangle(data.type().name()) +
              "'as arithmetic type .");
}

py::object cast_other(const any& data) {
  const auto& type = data.type().type();
  if (type == typeid(std::string)) {
    return py::bytes(py_cast<std::string>(data));
  }
#ifdef WITH_OPENCV
  else if (type == typeid(cv::Mat)) {
    cv::Mat tm = any_cast<cv::Mat>(data);
    return mat2numpy(tm);
  }
#endif
  else if (type == typeid(at::Tensor)) {
    return py_cast<at::Tensor>(data);
  } else if (type == typeid(std::shared_ptr<SimpleEvents>)) {
    return py_cast<std::shared_ptr<SimpleEvents>>(data);
  }
  return py::cast(data);
  // IPIPE_THROW(std::string("can not treat '") + local_demangle(data.type().name()) +
  //             "'as arithmetic type .");
}

py::object list2numpy(const any& data) {
  const auto& in = data.inner_type();
  if (in == typeid(int)) {
    const auto* pdata = any_cast<std::vector<int>>(&data);
    IPIPE_ASSERT(pdata);
    return list2numpy<int>(*pdata);
  } else if (in == typeid(float)) {
    const auto* pdata = any_cast<std::vector<float>>(&data);
    IPIPE_ASSERT(pdata);
    return list2numpy<float>(*pdata);
  } else if (in == typeid(double)) {
    const auto* pdata = any_cast<std::vector<double>>(&data);
    IPIPE_ASSERT(pdata);
    return list2numpy<double>(*pdata);
  } else if (in == typeid(unsigned int)) {
    const auto* pdata = any_cast<std::vector<unsigned int>>(&data);
    IPIPE_ASSERT(pdata);
    return list2numpy<unsigned int>(*pdata);
  } else if (in == typeid(char)) {
    const auto* pdata = any_cast<std::vector<char>>(&data);
    IPIPE_ASSERT(pdata);
    return list2numpy<char>(*pdata);
  } else if (in == typeid(unsigned char)) {
    const auto* pdata = any_cast<std::vector<unsigned char>>(&data);
    IPIPE_ASSERT(pdata);
    return list2numpy<unsigned char>(*pdata);
  }
  IPIPE_THROW("unsupported arithmetic types");
}

py::object list22numpy(const any& data) {
  const auto& in = data.inner_type();
  if (in == typeid(std::vector<int>)) {
    const auto* pdata = any_cast<std::vector<std::vector<int>>>(&data);
    IPIPE_ASSERT(pdata);
    return list22numpy<int>(*pdata);
  } else if (in == typeid(std::vector<float>)) {
    const auto* pdata = any_cast<std::vector<std::vector<float>>>(&data);
    IPIPE_ASSERT(pdata);
    return list22numpy<float>(*pdata);
  } else if (in == typeid(std::vector<double>)) {
    const auto* pdata = any_cast<std::vector<std::vector<double>>>(&data);
    IPIPE_ASSERT(pdata);
    return list22numpy<double>(*pdata);
  } else if (in == typeid(std::vector<unsigned int>)) {
    const auto* pdata = any_cast<std::vector<std::vector<unsigned int>>>(&data);
    IPIPE_ASSERT(pdata);
    return list22numpy<unsigned int>(*pdata);
  } else if (in == typeid(std::vector<char>)) {
    const auto* pdata = any_cast<std::vector<std::vector<char>>>(&data);
    IPIPE_ASSERT(pdata);
    return list22numpy<char>(*pdata);
  } else if (in == typeid(std::vector<unsigned char>)) {
    const auto* pdata = any_cast<std::vector<std::vector<unsigned char>>>(&data);
    IPIPE_ASSERT(pdata);
    return list22numpy<unsigned char>(*pdata);
  }
  IPIPE_THROW("unsupported arithmetic types");
}

py::object list22tensor(const any& data) {
  const auto& in = data.inner_type();
  if (in == typeid(std::vector<at::Tensor>)) {
    const auto* pdata = any_cast<std::vector<std::vector<at::Tensor>>>(&data);
    IPIPE_ASSERT(pdata);
    return py::cast(*pdata);
  }
  IPIPE_THROW("unsupported tensor types");
}

py::object list32numpy(const any& data) {
  const auto& in = data.inner_type();
  if (in == typeid(std::vector<std::vector<int>>)) {
    const auto* pdata = any_cast<std::vector<std::vector<std::vector<int>>>>(&data);
    IPIPE_ASSERT(pdata);
    return list32numpy<int>(*pdata);
  } else if (in == typeid(std::vector<std::vector<float>>)) {
    const auto* pdata = any_cast<std::vector<std::vector<std::vector<float>>>>(&data);
    IPIPE_ASSERT(pdata);
    return list32numpy<float>(*pdata);
  } else if (in == typeid(std::vector<std::vector<double>>)) {
    const auto* pdata = any_cast<std::vector<std::vector<std::vector<double>>>>(&data);
    IPIPE_ASSERT(pdata);
    return list32numpy<double>(*pdata);
  } else if (in == typeid(std::vector<std::vector<unsigned int>>)) {
    const auto* pdata = any_cast<std::vector<std::vector<std::vector<unsigned int>>>>(&data);
    IPIPE_ASSERT(pdata);
    return list32numpy<unsigned int>(*pdata);
  } else if (in == typeid(std::vector<std::vector<char>>)) {
    const auto* pdata = any_cast<std::vector<std::vector<std::vector<char>>>>(&data);
    IPIPE_ASSERT(pdata);
    return list32numpy<char>(*pdata);
  } else if (in == typeid(std::vector<std::vector<unsigned char>>)) {
    const auto* pdata = any_cast<std::vector<std::vector<std::vector<unsigned char>>>>(&data);
    IPIPE_ASSERT(pdata);
    return list32numpy<unsigned char>(*pdata);
  }
  IPIPE_THROW("unsupported arithmetic types");
}

template <template <class... Args> class container = std::unordered_set>
py::object cast_arithmetic_set(const any& data) {
  const auto& in = data.inner_type();
  if (in == typeid(int)) {
    const auto* pdata = any_cast<container<int>>(&data);
    IPIPE_ASSERT(pdata);
    return py::cast(*pdata);
  } else if (in == typeid(float)) {
    const auto* pdata = any_cast<container<float>>(&data);
    IPIPE_ASSERT(pdata);
    return py::cast(*pdata);
  } else if (in == typeid(double)) {
    const auto* pdata = any_cast<container<double>>(&data);
    IPIPE_ASSERT(pdata);
    return py::cast(*pdata);
  } else if (in == typeid(unsigned int)) {
    const auto* pdata = any_cast<container<unsigned int>>(&data);
    IPIPE_ASSERT(pdata);
    return py::cast(*pdata);
  } else if (in == typeid(char)) {
    const auto* pdata = any_cast<container<char>>(&data);
    IPIPE_ASSERT(pdata);
    return py::cast(*pdata);
  } else if (in == typeid(unsigned char)) {
    const auto* pdata = any_cast<container<unsigned char>>(&data);
    IPIPE_ASSERT(pdata);
    return py::cast(*pdata);
  } else if (in == typeid(bool)) {
    const auto* pdata = any_cast<container<bool>>(&data);
    IPIPE_ASSERT(pdata);
    return py::cast(*pdata);
  }
  IPIPE_THROW("unsupported arithmetic types");
}

py::object cast_other_set(const any& data) {
  const auto& in_t = data.inner_type();
  if (in_t == typeid(std::string)) {
    const auto* pdata = any_cast<std::unordered_set<std::string>>(&data);
    IPIPE_ASSERT(pdata);
    py::set result;
    for (const auto& item : *pdata) {
      result.add(py::bytes(item));
    }
    return result;
  }

  IPIPE_THROW("Unable to convert return value to Python object. inner type=" +
              local_demangle(in_t.name()));
}

py::object cast_other_str_dict(const any& data) {
  const auto& in_t = data.inner_type();
  if (in_t == typeid(std::string)) {
    const auto* pdata = any_cast<str_dict<std::string>>(&data);
    IPIPE_ASSERT(pdata);
    py::dict result;
    for (const auto& item : *pdata) {
      result[py::str(item.first)] = py::bytes(item.second);
    }
    return result;
  }
#ifdef WITH_OPENCV
  else if (in_t == typeid(cv::Mat)) {
    const auto* pdata = any_cast<str_dict<cv::Mat>>(&data);
    IPIPE_ASSERT(pdata);
    py::dict result;
    for (const auto& item : *pdata) {
      result[py::str(item.first)] = (mat2numpy(item.second));
    }
    return result;
  }
#endif
  else if (in_t == typeid(at::Tensor)) {
    const auto* pdata = any_cast<str_dict<at::Tensor>>(&data);
    IPIPE_ASSERT(pdata);
    return py::cast(*pdata);
  } else if (in_t == typeid(ipipe::any)) {
    const auto* pdata = any_cast<str_dict<ipipe::any>>(&data);
    IPIPE_ASSERT(pdata);
    py::dict result;
    for (const auto& item : *pdata) {
      if (item.first == TASK_DATA_KEY) {
        continue;
      }
      result[py::str(item.first)] = any2object(item.second);
    }
    return result;
  }
  IPIPE_THROW("Unable to convert return value to Python object. inner_type=" +
              local_demangle(in_t.name()));
}

py::object cast_other_list(const any& data) {
  const auto& in_t = data.inner_type();
  if (in_t == typeid(std::string)) {
    const auto* pdata = any_cast<std::vector<std::string>>(&data);
    IPIPE_ASSERT(pdata);
    py::list result;
    for (const auto& item : *pdata) {
      result.append(py::bytes(item));
    }
    return result;
  }
#ifdef WITH_OPENCV
  else if (in_t == typeid(cv::Mat)) {
    const auto* pdata = any_cast<std::vector<cv::Mat>>(&data);
    IPIPE_ASSERT(pdata);
    py::list result;
    for (const auto& item : *pdata) {
      result.append(mat2numpy(item));
    }
    return result;
  }
#endif
  else if (in_t == typeid(at::Tensor)) {
    const auto* pdata = any_cast<std::vector<at::Tensor>>(&data);
    IPIPE_ASSERT(pdata);
    return py::cast(*pdata);

    // for (const auto& item : *pdata) {
    //   result.append(item);
    // }
  } else if (in_t == typeid(ipipe::any)) {
    const auto* pdata = any_cast<std::vector<ipipe::any>>(&data);
    py::list result;
    IPIPE_ASSERT(pdata);
    for (const auto& item : *pdata) {
      result.append(any2object(item));
    }
    return result;

    // for (const auto& item : *pdata) {
    //   result.append(item);
    // }
  }
  IPIPE_THROW("Unable to convert return value to Python object.");
}
py::object any2object(const any& data) {
  std::vector<PyClassType> types = data.get_class_type();
  IPIPE_ASSERT(types.size() != 0);

  if (types.size() == 1) {
    switch (types[0]) {
      case PyClassType::arithmetic:
        return cast_arithmetic(data);
        break;
      case PyClassType::other:
        return cast_other(data);
        break;
      case PyClassType::unknown_container:
        IPIPE_THROW("empty python container is not allowed to return back to python. ");
        break;
      default:
        std::string error_mes = std::string("can not cast type '") +
                                local_demangle(data.type().name()) + "' to python " + ".";
        IPIPE_THROW(error_mes);
        break;
    }
  } else if (types.size() == 2) {
    if (types[0] == PyClassType::list) {
      if (types[1] == PyClassType::arithmetic) {
        return list2numpy(data);
      } else if (types[1] == PyClassType::other) {
        return cast_other_list(data);
      }
    } else if (types[0] == PyClassType::str_dict) {
      if (types[1] == PyClassType::arithmetic) {
        return cast_arithmetic_set<str_dict>(data);
      } else if (types[1] == PyClassType::other) {
        return cast_other_str_dict(data);
      }
    } else if (types[0] == PyClassType::set) {
      if (types[1] == PyClassType::arithmetic) {
        return cast_arithmetic_set<>(data);
      } else if (types[1] == PyClassType::other) {
        return cast_other_set(data);
      }
    }
  } else if (types.size() == 3) {
    if (types[0] == PyClassType::list && types[1] == PyClassType::list) {
      if (types[2] == PyClassType::arithmetic)
        return list22numpy(data);
      else if (types[2] == PyClassType::other)
        return list22tensor(data);
    }
  } else if (types.size() == 4) {
    if (types[0] == PyClassType::list && types[1] == PyClassType::list &&
        types[2] == PyClassType::list && types[3] == PyClassType::arithmetic) {
      return list32numpy(data);
    }
  }
  IPIPE_THROW("Unable to convert return value to Python object.");
}

}  // namespace ipipe

// // from https://github.com/pthom/cvnp/blob/master/cvnp/cvnp.cpp
// bool is_array_contiguous(const pybind11::array& a) {
//   pybind11::ssize_t expected_stride = a.itemsize();
//   for (int i = a.ndim() - 1; i >= 0; --i) {
//     pybind11::ssize_t current_stride = a.strides()[i];
//     if (current_stride != expected_stride) return false;
//     expected_stride = expected_stride * a.shape()[i];
//   }
//   return true;
// }