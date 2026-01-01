#include <optional>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "omniback/pybind/backend.hpp"

#include "omniback/builtin/basic_backends.hpp"
#include "omniback/core/backend.hpp"
#include "omniback/core/event.hpp"
#include "omniback/core/task_keys.hpp"
#include "omniback/pybind/converts.hpp"
#include "omniback/pybind/dict.hpp"
#include "omniback/pybind/py_helper.hpp"
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/macro.h"
// #include  "omniback/builtin/proxy.hpp"

namespace omniback {
using namespace omniback::python;

class OMNI_EXPORT PyInstance : public Backend {
 private:
  void impl_init(
      const std::unordered_map<string, string>& config,
      const dict& kwargs) override final {
    std::string init_method;
    py::gil_scoped_acquire gil;
    if (py::hasattr(*obj_, "init")) {
      init_method = "init";
      init_num_params_ =
          get_num_params(*obj_, init_method.data(), &init_default_params_);
    } else if (py::hasattr(*obj_, "__init__")) {
      init_method = "__init__";
      init_num_params_ =
          get_num_params(*obj_, init_method.data(), &init_default_params_);

    } else {
      SPDLOG_WARN(
          "No `init/__init__` method found in the python backend. Skip initialization.");
      return;
      throw std::invalid_argument(
          "No `init` method found in the python backend. You may need to "
          "use Forward[yourpython].");
    }
    SPDLOG_DEBUG(
        "(python)initialization, init_method = {}, init_num_params = {}, init_default_params = {} ",
        init_method,
        init_num_params_,
        init_default_params_);
    if (init_num_params_ > 0) {
      if (!kwargs) {
        if (init_num_params_ - init_default_params_ == 2)
          obj_->attr(init_method.data())(config);
        else if (init_num_params_ == 3 && init_default_params_ == 0) {
          obj_->attr(init_method.data())(config, py::none());
        } else if (init_num_params_ == 1 && init_default_params_ == 0) {
          obj_->attr(init_method.data())();
        } else {
          SPDLOG_WARN(
              "(python)Skip initialization, init_num_params = {}, init_default_params = {} ",
              init_num_params_,
              init_default_params_);
        }
        // throw std::invalid_argument(
        //     "init must have 1 or 2 arguments except self.");
      } else {
        if (init_num_params_ != 3)
          throw std::invalid_argument("init must have 2 arguments");
        obj_->attr(init_method.data())(config, PyDict(kwargs));
      }
    }
    // OMNI_ASSERT(!py::hasattr(*obj_, "min") && !py::hasattr(*obj_, "max"));
    max_ = this->Backend::max();
    min_ = this->Backend::min();
    if (py::hasattr(*obj_, "max")) {
      max_ = py::cast<size_t>(obj_->attr("max")());
    }
    if (py::hasattr(*obj_, "min")) {
      min_ = py::cast<size_t>(obj_->attr("min")());
    }
    SPDLOG_DEBUG("Python instance: min={} max={}", min_, max_);
  }
  void impl_forward(const std::vector<dict>& input_output) override final {
    std::vector<PyDict> py_input_output;
    for (const auto& item : input_output) {
      py_input_output.push_back(PyDict(item)); // no need gil
      // SPDLOG_DEBUG("py_input_output has result {}", item->find("result") !=
      // item->end());
    }
    {
      py::gil_scoped_acquire gil;
      try {
        obj_->attr("forward")(py_input_output);
      } catch (py::error_already_set& e) {
        std::string error_string = e.what();

        // 如果需要更详细的 traceback
        try {
          py::module_ traceback = py::module_::import("traceback");
          auto format_exc = traceback.attr("format_exc");
          std::string full_traceback = format_exc().cast<std::string>();

          SPDLOG_ERROR(
              "Python error: {}\nFull traceback:\n{}",
              error_string,
              full_traceback);
          throw std::runtime_error(
              "Python error: " + error_string + "\nTraceback:\n" +
              full_traceback);
        } catch (...) {
          // 如果 traceback 获取失败，至少记录基本错误信息
          SPDLOG_ERROR("Python error (traceback failed): {}", error_string);
          throw std::runtime_error("Python error: " + error_string);
        }
      }
    }

    // SPDLOG_DEBUG("after forward -> py_input_output has result {}",
    // input_output[0]->find("result") != input_output[0]->end());
#ifdef DEBUG
    for (const auto& item : input_output) {
      if (item->find(TASK_RESULT_KEY) == item->end()) {
        SPDLOG_DEBUG("find no result in io from python side.");
      }
    }
#endif
  }
  uint32_t impl_max() const override final {
    return max_;
  }
  uint32_t impl_min() const override final {
    return min_;
  }

 public:
  void init_with_obj(const py::object& obj, bool update_max_min = true) {
    obj_ = omniback::python::make_unique(obj);
    if (update_max_min) {
      max_ = this->Backend::max();
      min_ = this->Backend::min();
      if (py::hasattr(obj, "max")) {
        max_ = py::cast<size_t>(obj.attr("max")());
      }
      if (py::hasattr(obj, "min")) {
        min_ = py::cast<size_t>(obj.attr("min")());
      }
    }
  }

 private:
  omniback::python::unique_ptr<> obj_;
  size_t max_ = std::numeric_limits<uint32_t>::max();
  size_t min_ = 1;
  size_t init_default_params_ = 0;
  size_t init_num_params_ = 0;
};

} // namespace omniback
namespace omniback {
namespace py = pybind11;
using namespace pybind11::literals;

namespace {
static std::shared_ptr<Backend> create_backend_from_py(
    const std::string& class_name,
    py::object aspect_name = py::none()) {
  const std::string aspect_name_str =
      aspect_name.is_none() ? "" : py::cast<std::string>(aspect_name);
  py::gil_scoped_release guard;
  auto backend = std::shared_ptr<Backend>(create_backend(class_name).release());
  if (!aspect_name_str.empty()) {
    // OMNI_INSTANCE_REGISTER(Backend, aspect_name_str, backend);
    register_backend(aspect_name_str, backend);
  }
  return backend;
};

using ConfigVariant = std::variant<
    std::unordered_map<std::string, std::string>,
    std::unordered_map<std::string, int>,
    std::unordered_map<std::string, double>>;
std::unordered_map<std::string, std::string> convert_config(
    const std::optional<ConfigVariant>& config) {
  if (!config)
    return {};
  return std::visit(
      [](auto&& arg) {
        std::unordered_map<std::string, std::string> result;
        for (const auto& [key, value] : arg) {
          if constexpr (std::is_same_v<
                            std::decay_t<decltype(value)>,
                            std::string>) {
            result[key] = value;
          } else {
            result[key] = std::to_string(value);
          }
        }
        return result;
      },
      *config);
}

static std::shared_ptr<Backend> init_backend_from_py(
    const std::string& backend_config,
    std::optional<ConfigVariant> variant_config,
    std::optional<PyDict> kwargs_op,
    py::object aspect_name = py::none()) {
  auto config = convert_config(variant_config);
  const std::string aspect_name_str =
      aspect_name.is_none() ? "" : py::cast<std::string>(aspect_name);
  dict cpp_kwargs = kwargs_op ? (*kwargs_op).to_dict() : nullptr;
  py::gil_scoped_release guard;
  auto backend = std::shared_ptr<Backend>(
      init_backend(backend_config, config, cpp_kwargs).release());
  if (!aspect_name_str.empty()) {
    // OMNI_INSTANCE_REGISTER(Backend, aspect_name_str, backend);
    register_backend(aspect_name_str, backend);
  }
  return backend;
};

static void register_backend(
    const std::string& cls_name,
    std::function<Backend*()> f) {
  ClassRegistryInstance<Backend>().DoAddClass(cls_name, f);
}

static void register_cls(const std::string& cls_name, py::type obj) {
  auto creater = [obj, cls_name]() {
    auto* backend = new PyInstance();
    py::gil_scoped_acquire guard;
    // backend->init_with_obj(obj.attr("__new__")(obj));
    backend->init_with_obj(obj(), false);
    return (Backend*)backend;
  };

  register_backend(cls_name, creater);
}

static void register_backend(const std::string& aspect_name, py::object obj) {
  if (py::isinstance<py::type>(obj)) {
    register_cls(aspect_name, py::cast<py::type>(obj));
    return;
  }

  if (!py::hasattr(obj, "forward")) {
    if (py::isinstance<Backend>(obj)) {
      std::shared_ptr<Backend> backend =
          py::cast<std::shared_ptr<Backend>>(obj);
      // OMNI_INSTANCE_REGISTER(Backend, aspect_name, backend);
      register_backend(aspect_name, backend);
      return;
    } else {
      throw std::invalid_argument(
          "The backend must have a forward method, or is a "
          "omniback.Backend.");
    }
  }

  // do {
  //   if (py::isinstance<py::type>(obj)) {
  //     if (py::hasattr(obj, "__init__")) {
  //       size_t default_params = 0;
  //       int num_params = get_num_params(obj, "__init__", &default_params);
  //       if (num_params - default_params == 1) {
  //         obj = py::cast<py::type>(obj)();
  //         break;
  //       }
  //     }

  //     throw std::invalid_argument(
  //         "You must provide an instance, or a type name that can be "
  //         "default "
  //         "constructed (__init__(self)).");
  //   }
  // } while (false);

  int num_params = get_num_params(obj, "forward");
  OMNI_ASSERT(
      num_params == 2,
      "The forward function must have exactly one argument except self.");

  auto backend = std::make_shared<PyInstance>();
  backend->init_with_obj(obj);
  // std::unique_ptr<Backend> backend
  register_backend(aspect_name, backend);
  // OMNI_INSTANCE_REGISTER(Backend, aspect_name, backend);
}

void forward_backend(Backend& self, const py::object& input_output) {
  // SPDLOG_INFO("ref_count {}", input_output.ref_count());
  std::vector<dict> data;
  if (py::isinstance<PyDict>(input_output)) {
    data = {py::cast<PyDict&>(input_output).to_dict()};
  } else if (py::isinstance<py::list>(input_output)) {
    auto io_list = py::cast<py::list>(input_output);
    // Check if input_output is empty
    if (py::len(io_list) == 0) {
      throw std::invalid_argument("input list cannot be empty");
    }

    for (const auto& item : io_list) {
      // auto item_type = py::type::of(item);
      // std::cout << "Item type: "
      //           << item_type.attr("__name__").cast<std::string>() <<
      //           std::endl;

      OMNI_ASSERT(
          py::isinstance<PyDict>(item),
          " Unsupported input type. Please provide one of "
          "the following: dict, omniback.Dict, or "
          "List[omniback.Dict].");
      PyDict data_inside = py::cast<PyDict>(item);
      data.push_back(data_inside.to_dict());
    }
  } else if (py::isinstance<py::dict>(input_output)) {
    py::dict py_dict = py::cast<py::dict>(input_output);
    dict input = PyDict::py2dict(py_dict);
    if (input_output.contains(TASK_EVENT_KEY)) {
      throw py::key_error(
          std::string("The input dictionary contains key[`") + TASK_EVENT_KEY +
          "`]. This indicates an asynchronous call. In this "
          "case, writing "
          "back values to the Python dictionary is not "
          "supported yet. "
          "Please use omniback.Dict instead of dict as input.");
#if 0 // todo: add event support
                py::dict py_dict = py::cast<py::dict>(input_output);
                dict input = PyDict::py2dict(py_dict);
                data.push_back(input);
                auto pyevent = any_cast<Event>(input->at(TASK_EVENT_KEY));
                auto event = Event();
                (*input)[TASK_EVENT_KEY] = event;
                static const std::unordered_set<std::string> ignore_keys = {TASK_DATA_KEY};
                event->set_final_callback([input, py_dict, pyevent, event]() {
                  input->erase(TASK_EVENT_KEY);
                  py::gil_scoped_acquire gil_lock;
                  try{
                    PyDict::dict2py(input, py_dict, ignore_keys);
                    pyevent->notify_all_after_check_exception(event.get());
                  }
                  catch(...){
                    pyevent->set_exception_and_notify_all(std::current_exception());
                  }
    
                });
                py::gil_scoped_release rl;
                self.forward(data);
#endif
    }

    {
      py::gil_scoped_release rl;
      data.push_back(input);
      self.forward(data);
    }
    static const std::unordered_set<std::string> ignore_keys = {TASK_DATA_KEY};
    PyDict::dict2py(input, py_dict, ignore_keys);
    return;
  } else {
    throw std::invalid_argument(
        "unsupported input type. Try Union[dict, omniback.Dict, "
        "List[omniback.Dict]");
  }

  py::gil_scoped_release rl;
  self.forward(data);
}

} // namespace
void py_init_backend(py::module_& m) {
  m.def(
       "create",
       &create_backend_from_py,
       py::arg("name"),
       py::arg("register_name") = py::none(),
       R"pbdoc(
        Create a Backend object.

        Parameters:
            name (str): The name of the class to create.
            register_name (str, optional): The name of the registered aspect. Defaults to None.
        
        Returns:
            Backend: A new Backend object.
    )pbdoc")
      .def(
          "init",
          &init_backend_from_py,
          py::arg("backend"),
          py::arg("config") = py::none(),
          py::arg("kwargs") = py::none(),
          py::arg("register_name") = py::none(),
          R"pbdoc(
        Create and initialize a Backend object.

        Parameters:
            backend (str): Configuration string for the class to create. 
                          Supports bracket expansion rules, e.g., A[B,C] => {backend: A, "A::dependency": B}.
            config (dict): Configuration for the backend.
            register_name (str, optional): Name of the registered aspect. Defaults to None.
        
        Returns:
            Backend: A newly created Backend object.
    )pbdoc")
      .def(
          "unregister",
          &unregister_backend,
          py::arg("name"),
          R"pbdoc(
        Unregister a named backend instance.

        Parameters:
            name (str): The name of the backend to unregister.
    )pbdoc");
  m.def(
      "register",
      [](const std::string& name, const py::object& backend) {
        register_backend(name, backend);
      },
      py::arg("name"),
      py::arg("backend"),
      R"pbdoc(
        Register a named python backend instance (omniback.Backend, or a backend instance implemented the forward(self, List[omniback.Dict]) method).
        Usage: Forward[name] to execute the forward method only, or Init[name] to initialize the backend, or Launch to execute both.

        Parameters:
            name (str): The name to be registered.
            backend : A) An instance with a forward(self, [omniback.Dict]) method, or 
                      B) a default constructable <class 'type'> with a forward(self, [omniback.Dict]) method, or 
                      C) a omniback.Backend.
                 max()(default to maximum) and min()(default=1) method can alse be implemented.
                An init method can  be provided if to be used with Init.
    )pbdoc");
  // m.add_object("_cleanup", py::capsule([]() {
  // OMNI_INSTANCE_CLEANUP(Backend); }));
  m.def("get", &get_backend, py::return_value_policy::reference);

  auto atexit = py::module_::import("atexit");
  atexit.attr("register")(py::cpp_function([]() {
    py::gil_scoped_release gil;
    // OMNI_INSTANCE_CLEANUP(Backend);
    cleanup_backend();
  }));
  m.add_object("_registered_backend_cleanup", py::capsule([]() {
                 py::gil_scoped_release gil;
                 //  OMNI_INSTANCE_CLEANUP(Backend);
                 cleanup_backend();
               }));

  py::class_<Backend, std::shared_ptr<Backend>> omniback_backend(m, "Backend");

  omniback_backend.doc() =
      "omniback.backend provides an object wrapper for the "
      "omniback::Backend class";
  omniback_backend.def(
      "init",
      [](const std::shared_ptr<Backend>& self,
         str::str_map config,
         py::object kwargs = py::none()) {
        dict kwargs_dict = nullptr;

        // SPDLOG_INFO("init ptr = {}", (long long)&self);

        if (!kwargs.is_none()) {
          if (py::isinstance<py::dict>(kwargs)) {
            throw std::invalid_argument(
                "Unsupported type(<class 'dict'>) for kwargs. "
                "Please use omniback.Dict instead");
          }
          kwargs_dict = py::cast<const PyDict&>(kwargs).to_dict();
        }

        py::gil_scoped_release guard;

        // 调用 init 函数
        self->init(config, kwargs_dict);
        return self;
      },
      py::arg("config"),
      py::arg("kwargs") = py::none(),
      R"pbdoc(
            Initialize a Backend object.

            Parameters:
                config (dict): Configuration for the backend.
                kwargs (dict, optional): Shared configuration. Defaults to None.
        )pbdoc");

  omniback_backend.def(
      "as_function",
      [](std::shared_ptr<Backend> self) {
        return py::cpp_function(
            [self](const py::kwargs& kwargs) -> py::object {
              PyDict input_dict(
                  py::cast<py::dict>(kwargs)); // 需要先转换为py::dict

              dict cpp_dict = input_dict.to_dict();
              {
                py::gil_scoped_release release;
                self->forward({cpp_dict});
              }
              // {
              //   py::gil_scoped_release release;
              //   try {
              //     self->forward({cpp_dict});
              //   } catch (...) {
              //     py::gil_scoped_acquire acquire;
              //     throw py::error_already_set();
              //   }
              // }
              auto iter = cpp_dict->find(TASK_RESULT_KEY);
              OMNI_ASSERT(iter != cpp_dict->end());

              return any2object(iter->second);
            },
            py::keep_alive<0, 1>());
      },
      R"pbdoc(
              Convert the backend into a callable function that processes keyword arguments.
      
              Returns:
                  dict: Processed results. If the result contains the 'result' key, 
                       returns its value directly. Otherwise throw.
          )pbdoc");

  omniback_backend
      .def(
          "__call__",
          &forward_backend,
          py::arg("input_output"),
          R"pbdoc(
                Process input/output data. Support Union[dict, omniback.Dict, List[omniback.Dict]. The 'data' must be filled, and the results are stored in 'result' and other key-value pairs.
             )pbdoc")
      .def(
          "forward",
          [](Backend& self,
             const std::variant<
                 PyDict,
                 std::vector<PyDict>,
                 py::dict,
                 std::vector<py::dict>>& input_output,
             std::optional<Backend*> dep) {
            std::vector<dict> inputs;

            if (std::holds_alternative<PyDict>(input_output)) {
              inputs.push_back(std::get<PyDict>(input_output).to_dict());
            } else if (std::holds_alternative<py::dict>(input_output)) {
              auto data = std::get<py::dict>(input_output);
              inputs.push_back(PyDict::py2dict(data));
            } else if (std::holds_alternative<std::vector<PyDict>>(
                           input_output)) {
              auto data = std::get<std::vector<PyDict>>(input_output);
              for (auto& item : data) {
                inputs.push_back(item.to_dict());
              }
            } else if (std::holds_alternative<std::vector<py::dict>>(
                           input_output)) {
              auto data = std::get<std::vector<py::dict>>(input_output);
              for (auto& item : data) {
                inputs.push_back(PyDict::py2dict(item));
              }
            }
            py::gil_scoped_release rl;
            if (dep)
              return self.forward_with_dep(inputs, *(*dep));
            else
              return self.forward(inputs);
          },
          py::arg("input_output"),
          py::arg("dependency"),
          R"pbdoc(
                Process input/output data. Support Union[dict, omniback.Dict, List[omniback.Dict]. The 'data' must be filled, and the results are stored in 'result' and other key-value pairs.
             )pbdoc")
      .def("min", &Backend::min, R"pbdoc(
                Get the minimum size of the input/output data.
             )pbdoc")
      .def(
          "max",
          &Backend::max,
          R"pbdoc(
                Get the maximum size of the input/output data.
             )pbdoc");
}
} // namespace omniback