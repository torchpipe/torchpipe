#include "hami/core/any.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "hami/core/task_keys.hpp"
#include "hami/csrc/py_register.hpp"
#include <pybind11/numpy.h>

namespace hami {
namespace py = pybind11;

using namespace pybind11::literals;

class Bytes {
   public:
    // 默认构造函数
    Bytes() = default;

    // 从std::string构造
    Bytes(const std::string& str) : data_(str) {}

    // 从char*和长度构造
    Bytes(const char* data, size_t size) : data_(data, size) {}

    // 从Python bytes构造
    Bytes(py::bytes bytes_obj) { data_ = bytes_obj.cast<std::string>(); }

    // 获取底层数据
    const std::string& data() const { return data_; }

    // 转换为std::string
    std::string toString() const { return data_; }

    // 长度
    size_t size() const { return data_.size(); }

   private:
    std::string data_;  // 核心数据
};

}  // namespace hami

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
    static handle cast(const hami::Bytes& src, return_value_policy policy,
                       handle parent) {
        return py::bytes(src.data()).release();
    }
};
}  // namespace detail
}  // namespace pybind11

namespace hami {
void init_numpy(py::module_& m) {}

HAMI_ADD_HASH(Bytes);

class Str2Bytes : public BackendOne {
   public:
    void forward(const dict& data) : override final {
        data->insert_or_assign(
            TASK_RESULT_KEY,
            Bytes(any_cast<std::string>(data->at(TASK_DATA_KEY))));
    }
};

HAMI_REGISTER_BACKEND(Str2Bytes);

}  // namespace hami