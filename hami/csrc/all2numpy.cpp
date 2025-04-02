#include "hami/csrc/all2numpy.hpp"
#include "hami/core/any.hpp"

#include <pybind11/pybind11.h>

#include "hami/core/task_keys.hpp"
#include "hami/csrc/py_register.hpp"
#include <pybind11/numpy.h>

#include "hami/core/backend.hpp"
// PYBIND11_MAKE_OPAQUE(std::vector<int>);
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace pybind11 {
namespace detail {
namespace py = pybind11;

template <>
struct type_caster<std::vector<int>> {
 public:
  PYBIND11_TYPE_CASTER(std::vector<int>, _("numpy.ndarray"));

  static handle cast(
      const std::vector<int>& src,
      return_value_policy policy,
      handle parent) {
    if (policy == return_value_policy::take_ownership) {
      auto* vec_ptr = new std::vector<int>(std::move(src));
      py::capsule capsule(
          vec_ptr, [](void* v) { delete static_cast<std::vector<int>*>(v); });
      return py::array_t<int>(vec_ptr->size(), vec_ptr->data(), capsule)
          .release();
    } else {
      py::array_t<int> arr(src.size());
      std::memcpy(arr.mutable_data(), src.data(), src.size() * sizeof(int));
      return arr.release();
    }
  }
};

} // namespace detail
} // namespace pybind11

namespace hami {
namespace py = pybind11;

using namespace pybind11::literals;

// py::array_t<int> to_numpy(std::vector<int>&& vec) {
//   // 将vector移动到堆上
//   auto* vec_ptr = new std::vector<int>(std::move(vec));
//   // 创建胶囊以管理vector的生命周期
//   auto capsule = py::capsule(
//       vec_ptr, [](void* v) { delete reinterpret_cast<std::vector<int>*>(v);
//       });
//   // 创建NumPy数组并关联胶囊
//   return py::array_t<int>(
//       {vec_ptr->size()}, // 形状
//       {sizeof(int)}, // 步长
//       vec_ptr->data(), // 数据指针
//       capsule // 胶囊确保vector在不再使用时释放
//   );
// }

class Bytes {
 public:
  // 默认构造函数
  Bytes() = default;

  // 从std::string构造
  Bytes(const std::string& str) : data_(str) {}

  // 从char*和长度构造
  Bytes(const char* data, size_t size) : data_(data, size) {}

  // 从Python bytes构造
  Bytes(py::bytes bytes_obj) {
    data_ = bytes_obj.cast<std::string>();
  }

  // 获取底层数据
  const std::string& data() const {
    return data_;
  }

  // 转换为std::string
  std::string toString() const {
    return data_;
  }

  // 长度
  size_t size() const {
    return data_.size();
  }

 private:
  std::string data_; // 核心数据
};

} // namespace hami

// 添加类型转换支持
namespace pybind11 {
namespace detail {
template <>
struct type_caster<hami::Bytes> {
 public:
  PYBIND11_TYPE_CASTER(hami::Bytes, _("Bytes"));

  // Python -> C++ 转换
  bool load(handle src, bool convert) {
    if (py::isinstance<py::bytes>(src)) {
      value = hami::Bytes(py::cast<py::bytes>(src)).data();
      return true;
    }
    return false;
  }

  // C++ -> Python 转换
  static handle cast(
      const hami::Bytes& src,
      return_value_policy policy,
      handle parent) {
    return py::bytes(src.data()).release();
  }
};
} // namespace detail
} // namespace pybind11

namespace hami {
void init_numpy(py::module_& m) {}

HAMI_ADD_HASH(Bytes);

class Str2Bytes : public BackendOne {
 private:
  void forward(const hami::dict& data) override final {
    data->insert_or_assign(
        TASK_RESULT_KEY, Bytes(any_cast<std::string>(data->at(TASK_DATA_KEY))));
  }
};

HAMI_REGISTER_BACKEND(Str2Bytes);

} // namespace hami