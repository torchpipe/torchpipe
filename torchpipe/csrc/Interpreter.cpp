// Copyright 2021-2024 NetEase.
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

#include "Interpreter.hpp"
#include "base_logging.hpp"
#include "dict_helper.hpp"
#include "event.hpp"
#include "PipelineV3.hpp"
#include "BaselineSchedule.hpp"
#ifdef PYBIND
#include <pybind11/stl.h>
#include "Python.hpp"
#endif
namespace ipipe {
namespace {
std::unique_ptr<Backend> g_env;
}

bool Interpreter::init(const std::unordered_map<std::string, std::string>& config,
                       dict shared_config) {
  try {
    init(config);
  } catch (const std::exception& e) {
    SPDLOG_ERROR(e.what());
    return false;
  }
  return true;
}

void Interpreter::init(const std::unordered_map<std::string, std::string>& config) {
  mapmap global_config;
  global_config["global"] = config;
  init(global_config);
}

void Interpreter::init(mapmap config) {
  if (config.size() == 1) {
    const auto key = config.begin()->first;
    if (key != "global") {
      config["global"] = config[key];
      config.erase(key);
    }
  }
  if (config.empty()) throw std::invalid_argument("empty config");
  // 处理全局配置, 将全局配置放在global键值
  auto iter_global = config.find("global");
  if (iter_global != config.end()) {
    update_global(config);
  } else {
    config["global"] = std::unordered_map<std::string, std::string>();
    iter_global = config.find("global");
    if (config.size() == 1) {
      throw std::runtime_error("empty config. start failed");
    }
  }

  // todo 此处分散到engine中，而非集中在解释器设置
  handle_config(config);

  // Environment Initialization
  auto iter_logger = iter_global->second.find("Interpreter::env");
  std::string env_name;
  if (iter_logger != iter_global->second.end()) {
    // g_env = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, iter_logger->second));
    env_name = iter_logger->second;
  } else {
    // env_name = "";
    // g_env = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, "Logger"));
  }
  static bool benv_finish = false;
  {
    std::call_once(once_flag_, &Interpreter::env_init, this, iter_global->second, nullptr, env_name,
                   benv_finish);
  }

  if (!benv_finish) {
    throw std::runtime_error("Environment Initialization failed");
  }
  assert(g_env);

  // 获取主引擎
  auto iter_main = iter_global->second.find("Interpreter::backend");
  /// 如果没有配置主引擎，则根据节点数目选择默认主程序
  if (iter_main == iter_global->second.end() || iter_main->second.empty()) {
    if (config.size() <= 2) {
      // 如果是单节点, 默认启动 BaselineSchedule 引擎进行调度
      backend_ = std::make_unique<BaselineSchedule>();
    } else {
      // 如果是多节点， 默认启动 PipelineV3 引擎进行 管理
      backend_ = std::make_unique<PipelineV3>();
    }
  }
  // 如果配置了主引擎，根据配置创建主引擎。
  else {
    SPDLOG_INFO("main engine: {}", iter_main->second);
    backend_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, iter_main->second));
  }

  // 初始化主引擎
  dict shared_dict = std::make_shared<std::unordered_map<std::string, any>>();
  (*shared_dict)["config"] = config;
  std::unordered_map<std::string, std::string> static_config;
  if (config.size() == 1) {
    static_config = config["global"];
  } else if (config.size() == 2) {
    for (auto iter_config = config.begin(); iter_config != config.end(); ++iter_config) {
      if (iter_config->first == "global") continue;
      static_config = iter_config->second;
      auto iter_node_name = static_config.find("node_name");
      if (iter_node_name != static_config.end()) {
        if (iter_node_name->second != iter_config->first) {
          SPDLOG_ERROR("node_name not match: " + iter_node_name->second + " vs. " +
                       iter_config->first);
        }
      } else {
        static_config["node_name"] = iter_config->first;
      }
    }
  }
  (*shared_dict)["Interpreter"] = (Backend*)this;
  if (!backend_ || !backend_->init(static_config, shared_dict)) {
    backend_ = nullptr;
    throw std::runtime_error("Interpreter init failed");
  }
  SPDLOG_INFO("Initialization completed.");
}

uint32_t Interpreter::max() const { return backend_->max(); }

uint32_t Interpreter::min() const { return backend_->min(); }

void Interpreter::env_init(const std::unordered_map<std::string, std::string>& config,
                           dict dict_config, const std::string& backend_name, bool& finished) {
  finished = false;

  if (!backend_name.empty()) {
    g_env = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, backend_name));
  } else {
    g_env = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, "Logger"));
  }

  try {
    finished = g_env->init(config, dict_config);
  } catch (std::exception& e) {
    SPDLOG_ERROR("env init failed: {}", e.what());
    finished = false;
  }
}

#ifdef PYBIND

void Interpreter::forward(py::list py_inputs) {
  std::vector<dict> inputs;

  for (std::size_t i = 0; i < py::len(py_inputs); ++i) {
    IPIPE_ASSERT(py::isinstance<py::dict>(py_inputs[i]));
    inputs.push_back(py2dict(py_inputs[i]));
  }

  {
    py::gil_scoped_release gil_lock;
    forward(inputs);
  }

  for (std::size_t i = 0; i < py::len(py_inputs); ++i) {
    dict2py(inputs[i], py_inputs[i]);
  }

  return;
}
void Interpreter::forward(py::dict py_input) {
  dict input = py2dict(py_input);

  auto iter = input->find(TASK_EVENT_KEY);
  if (iter != input->end()) {
    std::shared_ptr<SimpleEvents> input_event =
        any_cast<std::shared_ptr<SimpleEvents>>(iter->second);

    if (!input_event->valid())
      // input_event is invalid, so we cannot use it to store the exception;
      throw py::value_error("Invalid input event. Maybe it has been used before.");
    // add_callback need gil lock to handle py_input
    input_event->add_callback([input, py_input]() {
      py::gil_scoped_acquire gil_lock;
      dict2py(input, py_input);
    });
    py::gil_scoped_release gil_lock;
    forward({input});
    return;
  }
  {
    py::gil_scoped_release gil_lock;
    forward({input});
  }
  dict2py(input, py_input);
}
#endif

void Interpreter::forward(const std::vector<dict>& input_dicts) {
  if (!backend_) throw std::runtime_error("not initialized");
  for (const auto& da : input_dicts) {
    auto iter = da->find(TASK_DATA_KEY);
    if (iter == da->end()) {
      for (auto z : *da) {
        SPDLOG_INFO("key: {}", z.first);
      }
      throw std::out_of_range("TASK_DATA_KEY (data) not exists");
    }
    da->erase(TASK_RESULT_KEY);
  }

  backend_->forward(input_dicts);
}

std::once_flag Interpreter::once_flag_;

#ifdef PYBIND  // pyhton binding
void encrypt_file_to_file(std::string file_path, std::string out_file_path, std::string key);

#ifdef WITH_TENSORRT
void init_infer_shape(py::module& m);
void supported_opset(py::module& m);
#endif

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

  // bind_backend<MyBackend1>(m, "MyBackend1");
  // bind_backend<MyBackend2>(m, "MyBackend2");
  // py::class_<AnyWrapper>(m, "Any").def(py::init<>());
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
  init_infer_shape(m);
  supported_opset(m);
#endif
}
#endif

}  // namespace ipipe