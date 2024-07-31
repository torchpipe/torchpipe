
#ifdef PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#if PYBIND11_VERSION_MAJOR == 2
#if PYBIND11_VERSION_MINOR < 7
#error "This code requires pybind11 version 2.7 or greater"
#endif
#endif
#endif

#ifdef PYBIND

// base headers
#include "any.hpp"
#include "threadsafe_kv_storage.hpp"
#include "Interpreter.hpp"
#include "event.hpp"
#include "filter.hpp"
#include "infer_model_input_shape.hpp"

// python related
#include "tensor_type_caster.hpp"
#include "Python.hpp"
#include "any2object.hpp"
#include "object2any.hpp"

namespace pybind11 {
namespace detail {
template <>
struct type_caster<ipipe::any> {
 public:
  PYBIND11_TYPE_CASTER(ipipe::any, _("Any"));

  // Python -> C++
  bool load(handle src, bool) {
    value = ipipe::object2any(src);
    return true;
  }

  // C++ -> Python
  static handle cast(const ipipe::any& src, return_value_policy /* policy */, handle /* parent */) {
    return ipipe::any2object(src).release();
  }
};

template <>
struct type_caster<ipipe::dict> {
 public:
  PYBIND11_TYPE_CASTER(ipipe::dict, _("ipipe::dict"));

  // Python -> C++
  bool load(handle src, bool) {
    auto dict = src.cast<py::dict>();
    auto map = std::make_shared<std::unordered_map<std::string, ipipe::any>>();
    for (auto item : dict) {
      (*map)[item.first.cast<std::string>()] = ipipe::object2any(item.second);
    }
    value = map;
    return true;
  }

  // C++ -> Python
  static handle cast(const ipipe::dict& src, return_value_policy /* policy */,
                     handle /* parent */) {
    py::dict dict;
    for (const auto& item : *src) {
      dict[item.first.c_str()] = ipipe::any2object(item.second);
    }
    return dict.release();
  }
};

}  // namespace detail
}  // namespace pybind11

namespace ipipe {

#ifdef PYBIND  // pyhton binding
void encrypt_file_to_file(std::string file_path, std::string out_file_path, std::string key);

#ifdef WITH_CUDA
std::string get_sm();
#endif
std::vector<std::string> list_backends() { return IPIPE_ALL_NAMES(ipipe::Backend); }
bool is_registered(std::string backend) {
  static std::vector<std::string> all_backens = IPIPE_ALL_NAMES(ipipe::Backend);
  return std::find(all_backens.begin(), all_backens.end(), backend) != all_backens.end();
}

template <typename T>
void bind_backend(py::module& m, const char* name) {
  py::class_<T, Backend, std::shared_ptr<T>>(m, name)
      .def(py::init<>())
      .def("init", &T::init, py::arg("config"),
           py::arg_v("dict_config", py::none(), "optional dictionary config"),
           py::call_guard<py::gil_scoped_release>())
      .def("forward", &T::forward, py::arg("input_dicts"), py::call_guard<py::gil_scoped_release>())
      .def("max", &T::max)
      .def("min", &T::min);
}

void register_backend(py::object class_def, const std::string& register_name) {
  register_py(class_def, register_name);
}

void register_filter(py::object class_def, const std::string& register_name) {
  register_py_filter(class_def, register_name);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("TASK_RESULT_KEY") = py::cast(TASK_RESULT_KEY);
  m.attr("TASK_DATA_KEY") = py::cast(TASK_DATA_KEY);
  m.attr("TASK_BOX_KEY") = py::cast(TASK_BOX_KEY);
  m.attr("TASK_INFO_KEY") = py::cast(TASK_INFO_KEY);
  m.attr("TASK_EVENT_KEY") = py::cast(TASK_EVENT_KEY);
  m.attr("TASK_NODE_NAME_KEY") = py::cast(TASK_NODE_NAME_KEY);

#ifdef WITH_TENSORRT
  m.attr("WITH_TENSORRT") = py::cast(true);
#else
  m.attr("WITH_TENSORRT") = py::cast(false);
#endif

#ifdef WITH_CUDA
  m.attr("WITH_CUDA") = py::cast(true);
  m.def("get_sm", &get_sm, py::call_guard<py::gil_scoped_release>());
#else
  m.attr("WITH_CUDA") = py::cast(false);
#endif
#ifdef WITH_OPENVINO
  m.attr("WITH_OPENVINO") = py::cast(true);
#else
  m.attr("WITH_OPENVINO") = py::cast(false);
#endif

  m.def("parse_toml", &parse_toml, py::call_guard<py::gil_scoped_release>(), py::arg("path_toml"));
  m.def("encrypt", &encrypt_file_to_file, py::call_guard<py::gil_scoped_release>(),
        py::arg("file_path"), py::arg("out_file_path"), py::arg("key"));

  m.def("list_backends", &list_backends, py::call_guard<py::gil_scoped_release>());
  m.def("is_registered", &is_registered, py::call_guard<py::gil_scoped_release>());

  m.def("register_backend", &register_backend, py::arg("class_def"), py::arg("register_name"));
  m.def("register_filter", &register_filter, py::arg("class_def"), py::arg("register_name"));

  /**
   * @brief c++ Interpreter wrapper
   */
  pybind11::class_<Interpreter> nodes_wrapper(m, "Interpreter");
  nodes_wrapper
      .def(py::init(), py::return_value_policy::take_ownership,
           R"docdelimiter(
                    create an Interpreter.

                    Parameters:
                )docdelimiter")
      .def("init_from_toml", &Interpreter::init_from_toml, py::call_guard<py::gil_scoped_release>(),
           py::return_value_policy::take_ownership,
           R"docdelimiter(
                    Initialize Interpreter from toml file

                    Parameters:
                        toml_path: string 
                    Returns:
                        bool.
                )docdelimiter",
           py::arg("toml_path"))
      .def("init", py::overload_cast<mapmap>(&Interpreter::init),
           py::call_guard<py::gil_scoped_release>(), py::return_value_policy::take_ownership,
           R"docdelimiter(
                    Initialize Interpreter from dict-of-dict

                    Parameters:
                        config: dict-of-dict
                    Returns:
                        bool.
                )docdelimiter",
           py::arg("config"))
      .def("init",
           py::overload_cast<const std::unordered_map<std::string, std::string>&>(
               &Interpreter::init),
           py::call_guard<py::gil_scoped_release>(), py::return_value_policy::take_ownership,
           R"docdelimiter(
                    Initialize Interpreter from  dict of string

                    Parameters:
                        config: dict
                    Returns:
                        bool.
                )docdelimiter",
           py::arg("config"))
      .def("max", &Interpreter::max, py::call_guard<py::gil_scoped_release>(),
           py::return_value_policy::take_ownership,
           R"docdelimiter(
                    return Interpreter'max()

                    Returns:
                        int.
                )docdelimiter")
      .def("min", &Interpreter::min, py::call_guard<py::gil_scoped_release>(),
           py::return_value_policy::take_ownership,
           R"docdelimiter(
                    return Interpreter'min()
                    
                    Returns:
                        int.
                )docdelimiter")
      .def("__call__", py::overload_cast<py::list>(&Interpreter::forward),
           py::return_value_policy::take_ownership,
           R"docdelimiter(
                    forward

                    Parameters:
                        data: List[Dict]

                    Returns:
                        None

                )docdelimiter",
           py::arg("data"))
      .def("__call__", py::overload_cast<py::dict>(&Interpreter::forward),
           py::return_value_policy::take_ownership,
           R"docdelimiter(
                    forward

                    Parameters:
                        data: Dict

                    Returns:
                        None
                )docdelimiter",
           py::arg("data"));

  //   py::class_<ThreadSafeKVStorage>(m, "ThreadSafeKVStorage")
  //       .def("read",
  //            [](ThreadSafeKVStorage& self, const std::string& path) {
  //              py::dict py_input;
  //              ipipe::dict2py(self.read(path), py_input, true);
  //              return py_input;
  //            })
  //       .def("write", [](ThreadSafeKVStorage& self, const std::string& path,
  //                        py::dict data) { self.write(path, py2dict(data)); })
  //       .def("clear", py::overload_cast<>(&ThreadSafeKVStorage::clear))
  //       .def("clear", py::overload_cast<const std::string&>(&ThreadSafeKVStorage::clear))
  //       .def_static("getInstance", &ThreadSafeKVStorage::getInstance,
  //                   py::return_value_policy::reference);

  py::class_<ipipe::any>(m, "Any").def(py::init<>());

  py::enum_<Filter::status>(m, "Status")
      .value("Run", Filter::status::Run)
      .value("Skip", Filter::status::Skip)
      .value("SerialSkip", Filter::status::SerialSkip)
      .value("SubGraphSkip", Filter::status::SubGraphSkip)
      .value("Break", Filter::status::Break)
      .value("Error", Filter::status::Error);

  pybind11::class_<SimpleEvents, std::shared_ptr<SimpleEvents>>(m, "Event")
      .def(py::init<>())
      .def("Wait", &SimpleEvents::Wait, py::call_guard<py::gil_scoped_release>(),
           R"docdelimiter(
                    wait for exception or finish

                    Parameters:
                        
                    Returns:
                        None.
                )docdelimiter")
      .def("time_passed", &SimpleEvents::time_passed, py::call_guard<py::gil_scoped_release>(),
           R"docdelimiter(
                    check time used

                    Parameters:
                        void
                    Returns:
                        float.
                )docdelimiter");
#ifdef WITH_TENSORRT
  m.def("infer_shape", py::overload_cast<const std::string&>(&infer_shape),
        py::call_guard<py::gil_scoped_release>(), py::arg("model_path"));

  m.def("supported_opset", py::overload_cast<>(&supported_opset),
        py::call_guard<py::gil_scoped_release>());
#endif
}
#endif
}  // namespace ipipe
#endif