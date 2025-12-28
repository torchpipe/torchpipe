#include <omniback/pybind/python.hpp>
#include <omniback/extension.hpp>
#include <torch/csrc/autograd/python_variable.h>
#include <optional>

#include <ATen/ATen.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "pybind/register_types.h"

namespace torchpipe {
namespace py = pybind11;

// Helper function to check if an object is a PyTorch tensor
bool is_torch_tensor(const py::handle& obj) {
  //   static py::handle torch = py::module::import("torch").attr("Tensor");
  //   return py::isinstance(obj, torch);
  return THPVariable_Check(obj.ptr());
}

OMNI_ADD_PY2CPP([](const py::handle& obj) -> std::optional<omniback::any> {
  if (!is_torch_tensor(obj)) {
    return std::nullopt;
  }

  return omniback::any(py::cast<torch::Tensor>(obj));
});

OMNI_ADD_PY2CPP([](const py::handle& obj) -> std::optional<omniback::any> {
  if (!py::isinstance<py::list>(obj)) {
    return std::nullopt;
  }

  py::list py_list = py::cast<py::list>(obj);

  if (py_list.empty()) {
    return std::nullopt;
  }

  if (!is_torch_tensor(py_list[0])) {
    return std::nullopt;
  }

  std::vector<torch::Tensor> tensor_list;
  for (const auto& item : py_list) {
    if (!is_torch_tensor(item)) {
      return std::nullopt; // 如果列表中有一个元素不是张量，整个转换失败
    }
    tensor_list.push_back(py::cast<torch::Tensor>(item));
  }

  return omniback::any(tensor_list);
});

OMNI_ADD_HASH(torch::Tensor);
OMNI_ADD_HASH(std::vector<torch::Tensor>)([](const omniback::any& data) {
  const auto& vec = omniback::any_cast<std::vector<torch::Tensor>>(data);
  py::list result;
  for (const auto& tensor : vec) {
    result.append(py::cast(tensor));
  }
  return result;
});

OMNI_ADD_CPP2PY([](const omniback::any& obj) -> std::optional<py::object> {
  if (obj.type() != typeid(torch::Tensor)) {
    return std::nullopt;
  }

  const torch::Tensor& tensor = omniback::any_cast<torch::Tensor>(obj);
  return py::cast(tensor);
});
// Add this to your initialization code
} // namespace torchpipe