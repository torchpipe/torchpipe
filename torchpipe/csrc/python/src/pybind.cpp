
#ifdef PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/stl_bind.h>

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
#include "threadsafe_queue.hpp"

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
    // return py::cast(data).release();
  }
};

// template <>
// struct type_caster<ipipe::dict> {
//  public:
//   PYBIND11_TYPE_CASTER(ipipe::dict, _("ipipe::dict"));

//   // Python -> C++
//   bool load(handle src, bool) {
//     auto dict = src.cast<py::dict>();
//     auto map = std::make_shared<std::unordered_map<std::string, ipipe::any>>();
//     for (auto item : dict) {
//       (*map)[item.first.cast<std::string>()] = ipipe::object2any(item.second);
//     }
//     value = map;
//     return true;
//   }

//   // C++ -> Python
//   static handle cast(const ipipe::dict& src, return_value_policy /* policy */,
//                      handle /* parent */) {
//     py::dict dict;
//     for (const auto& item : *src) {
//       dict[item.first.c_str()] = ipipe::any2object(item.second);
//     }
//     return dict.release();
//   }
// };

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

template <typename T>
void bind_threadsafe_queue(py::module& m, const std::string& typestr) {
  using Queue = ipipe::ThreadSafeQueue<T>;
  std::string pyclass_name = std::string("ThreadSafeQueue") + typestr;
  py::class_<Queue, std::shared_ptr<Queue>>(m, pyclass_name.c_str())
      .def(py::init<>())
      .def("Push", py::overload_cast<const T&>(&Queue::Push),
           py::call_guard<py::gil_scoped_release>())
      .def("Push", py::overload_cast<const std::vector<T>&>(&Queue::Push),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "WaitForPop",
          [](Queue& q, int time_out) -> py::object {
            T value;
            bool result = q.WaitForPop(value, time_out);
            py::gil_scoped_acquire local_guard;
            if (result) {
              return py::cast(value);
            } else {
              return py::none();
            }
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "pop_all",
          [](Queue& q) {
            std::vector<T> all_data = q.PopAll();
            py::gil_scoped_acquire local_guard;
            return py::cast(all_data);
          },
          py::call_guard<py::gil_scoped_release>())
      .def("empty", &Queue::empty, py::call_guard<py::gil_scoped_release>())
      .def("size", &Queue::size, py::call_guard<py::gil_scoped_release>());
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
  m.def("unregister", &unregister);

  m.add_object("_cleanup", py::capsule(unregister));
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

  py::class_<ipipe::any>(m, "Any")
      .def(py::init<>())
      .def("as_str", [](ipipe::any& self) { return any_cast<std::string>(self); })
      .def("as_bytes",
           [](ipipe::any& self) {
             const std::string* result = any_cast<std::string>(&self);
             if (result) {
               return py::bytes(*result);
             } else {
               std::string tmp = any_cast<std::string>(self);  // let it throw
               return py::bytes(tmp);
             }
           })
      .def("as_int", [](ipipe::any& self) { return any_cast<int>(self); })
      .def("as_float", [](ipipe::any& self) { return any_cast<float>(self); })
      .def("as_double", [](ipipe::any& self) { return any_cast<double>(self); })
      .def("as_bool", [](ipipe::any& self) { return any_cast<bool>(self); })
      .def("as_vector_float", [](ipipe::any& self) { return any_cast<std::vector<float>>(self); })
      .def("as_vector_int", [](ipipe::any& self) { return any_cast<std::vector<int>>(self); })
      .def("as_vector_str",
           [](ipipe::any& self) { return any_cast<std::vector<std::string>>(self); })
      .def("cast",
           [](ipipe::any& self) {
             if (self.type().type() == typeid(std::string))
               return py::object(py::bytes(any_cast<std::string>(self)));
             return any2object(self);
           })
      .def("as_queue", [](ipipe::any& self) {
        if (typeid(std::shared_ptr<ThreadSafeQueue<long>>) == self.type()) {
          return py::cast(any_cast<std::shared_ptr<ThreadSafeQueue<long>>>(self));
        } else if (typeid(std::shared_ptr<ThreadSafeQueue<int>>) == self.type()) {
          return py::cast(any_cast<std::shared_ptr<ThreadSafeQueue<int>>>(self));
        } else if (typeid(std::shared_ptr<ThreadSafeQueue<float>>) == self.type()) {
          return py::cast(any_cast<std::shared_ptr<ThreadSafeQueue<float>>>(self));
        } else if (typeid(std::shared_ptr<ThreadSafeQueue<double>>) == self.type()) {
          return py::cast(any_cast<std::shared_ptr<ThreadSafeQueue<double>>>(self));
        } else if (typeid(std::shared_ptr<ThreadSafeQueue<short>>) == self.type()) {
          return py::cast(any_cast<std::shared_ptr<ThreadSafeQueue<short>>>(self));
        } else if (typeid(std::shared_ptr<ThreadSafeQueue<unsigned int>>) == self.type()) {
          return py::cast(any_cast<std::shared_ptr<ThreadSafeQueue<unsigned int>>>(self));
        } else {
          throw py::type_error(
              std::string("The object is not a std::shared_ptr<ThreadSafeQueue<T>>, is ") +
              self.type().name());
        }
      });

  {
    bind_threadsafe_queue<int>(m, "Int");
    bind_threadsafe_queue<float>(m, "Float");
    bind_threadsafe_queue<double>(m, "Double");
    bind_threadsafe_queue<long>(m, "Long");
    bind_threadsafe_queue<short>(m, "Short");
    bind_threadsafe_queue<unsigned int>(m, "UnsignedInt");

    py::class_<ThreadSafeDict, std::shared_ptr<ThreadSafeDict>>(m, "ThreadSafeDict")
        .def(py::init<>())
        .def("__getitem__",
             [](ThreadSafeKVStorage& self, const std::string& path, const std::string& key) {
               auto result = self.get(path, key);
               if (result) {
                 return ipipe::any2object(*result);
               } else {
                 return py::object(py::none());
               }
             })
        .def("__setitem__", [](ThreadSafeDict& self, const std::string& key,
                               const py::object& value) { self.set(key, object2any(value)); });

    py::class_<ThreadSafeKVStorage>(m, "ThreadSafeKVStorage")
        .def("__getitem__",
             [](ThreadSafeKVStorage& self, const std::pair<std::string, std::string>& key_pair) {
               //  {
               //    py::gil_scoped_release local_guard;
               //    result = self.get(path, key);
               //  }

               auto result = self.get(key_pair.first, key_pair.second);
               if (result) {
                 return ipipe::any2object(*result);
               } else {
                 return py::object(py::none());
               }
             })
        .def("__setitem__",
             [](ThreadSafeKVStorage& self, const std::string& path, const std::string& key,
                pybind11::handle data) { self.set_and_notify(path, key, object2any(data)); })
        .def("clear", py::overload_cast<>(&ThreadSafeKVStorage::clear))
        .def("erase", py::overload_cast<const std::string&>(&ThreadSafeKVStorage::erase))
        .def_static("getInstance", &ThreadSafeKVStorage::getInstance,
                    py::return_value_policy::reference);
  }

  py::enum_<Filter::status>(m, "Status")
      .value("Run", Filter::status::Run)
      .value("Skip", Filter::status::Skip)
      .value("SerialSkip", Filter::status::SerialSkip)
      .value("SubGraphSkip", Filter::status::SubGraphSkip)
      .value("Break", Filter::status::Break)
      .value("Error", Filter::status::Error);

  pybind11::class_<SimpleEvents, std::shared_ptr<SimpleEvents>>(m, "Event")
      // .def(py::init<>())
      .def(py::init<uint32_t>(), py::arg("num") = 1)
      .def("Wait", py::overload_cast<>(&SimpleEvents::Wait),
           py::call_guard<py::gil_scoped_release>(),
           R"docdelimiter(
                      wait for exception or finish

                      Parameters:
                        
                      Returns:
                        None.
                    )docdelimiter")
      .def("Wait", py::overload_cast<uint32_t>(&SimpleEvents::Wait), py::arg("timeout_ms"),
           py::call_guard<py::gil_scoped_release>(),
           R"docdelimiter(
                      wait for exception or finish with a timeout

                      Parameters:
                        timeout_ms (uint32_t): The timeout in milliseconds.
                        
                      Returns:
                        bool: True if the event is set before the timeout, false otherwise.
                    )docdelimiter")
      .def("time_passed", &SimpleEvents::time_passed, py::call_guard<py::gil_scoped_release>(),
           R"docdelimiter(
                    check time used

                    Parameters:
                        void
                    Returns:
                        float.
                )docdelimiter")
      .def("set_callback", &SimpleEvents::set_callback, py::call_guard<py::gil_scoped_release>(),
           py::arg("callback"))
      .def("set_final_callback", &SimpleEvents::set_final_callback,
           py::call_guard<py::gil_scoped_release>(), py::arg("callback"))
      .def("get_exception", &SimpleEvents::get_exception, py::call_guard<py::gil_scoped_release>())
      .def("try_throw", &SimpleEvents::try_throw, py::call_guard<py::gil_scoped_release>());

  py::class_<CustomDict>(m, "Dict")
      .def(py::init<>())
      .def("__getitem__",
           [](const CustomDict& d, const std::string& key) {
             auto result = d->find(key);
             if (result == d->end()) throw py::key_error("not found: " + key);
             return result->second;
           })
      .def("__setitem__", [](CustomDict& d, const std::string& key,
                             const ipipe::any& value) { d->insert({key, value}); })
      .def("__delitem__", [](CustomDict& d, const std::string& key) { d->erase(key); })
      .def("__contains__",
           [](const CustomDict& d, const std::string& key) { return d->find(key) != d->end(); })
      .def("__len__", [](const CustomDict& d) { return d->size(); })
      .def(
          "__iter__",
          [](const CustomDict& d) { return py::make_key_iterator(d->begin(), d->end()); },
          py::keep_alive<0, 1>())  // Keep object alive while iterator exists
      .def(
          "keys", [](const CustomDict& d) { return py::make_key_iterator(d->begin(), d->end()); },
          py::keep_alive<0, 1>())  // Keep object alive while iterator exists
      .def(
          "values",
          [](const CustomDict& d) { return py::make_value_iterator(d->begin(), d->end()); },
          py::keep_alive<0, 1>())  // Keep object alive while iterator exists
      .def(
          "items", [](const CustomDict& d) { return py::make_iterator(d->begin(), d->end()); },
          py::keep_alive<0, 1>());  // Keep object alive while iterator exists

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