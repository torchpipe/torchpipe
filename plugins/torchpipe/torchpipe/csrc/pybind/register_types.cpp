#include <hami/python.hpp>
#include <hami/extension.hpp>
#include <torch/csrc/autograd/python_variable.h>

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <ATen/ATen.h>
#include "pybind/register_types.h"

namespace torchpipe {
namespace py = pybind11;

// Helper function to check if an object is a PyTorch tensor
bool is_torch_tensor(const py::handle& obj) {
    //   static py::handle torch = py::module::import("torch").attr("Tensor");
    //   return py::isinstance(obj, torch);
    return THPVariable_Check(obj.ptr());
}

HAMI_ADD_PY2CPP([](const py::handle& obj) -> std::optional<hami::any> {
    if (!is_torch_tensor(obj)) {
        return std::nullopt;
    }

    return hami::any(py::cast<torch::Tensor>(obj));
});

HAMI_ADD_HASH(torch::Tensor);

HAMI_ADD_CPP2PY([](const hami::any& obj) -> std::optional<py::object> {
    if (obj.type() != typeid(torch::Tensor)) {
        return std::nullopt;
    }

    const torch::Tensor& tensor = hami::any_cast<torch::Tensor>(obj);
    return py::cast(tensor);
});
// Add this to your initialization code
}  // namespace torchpipe