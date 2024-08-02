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
#include <pybind11/pybind11.h>
#include "tensor_type_caster.hpp"
#endif

namespace ipipe {
namespace {

std::unique_ptr<Backend> g_env;

}  // namespace

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
  // 1. update global
  // 2. update node_name
  // 3. write config to shared_dict
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
      if (iter_global->second.find("scheduler") != iter_global->second.end()) {
        std::string tmp_name = any_cast<std::string>(iter_global->second.find("scheduler")->second);
        SPDLOG_DEBUG("use batching backend: {}", tmp_name);
        backend_ = std::unique_ptr<Backend>(IPIPE_CREATE(Backend, tmp_name));
      } else {
        // 如果是单节点, 默认启动 BaselineSchedule 引擎进行调度
        backend_ = std::make_unique<BaselineSchedule>();
      }

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

  std::unordered_map<std::string, std::string> static_config;
  if (config.size() == 1) {
    static_config = config["global"];
    config[TASK_DEFAULT_NAME_KEY] = config["global"];
    // config.erase("global");
  } else if (config.size() == 2) {
    for (auto iter_config = config.begin(); iter_config != config.end(); ++iter_config) {
      if (iter_config->first == "global") continue;
      static_config = iter_config->second;
      auto iter_node_name = static_config.find("node_name");
      if (iter_node_name != static_config.end()) {
        if (iter_node_name->second != iter_config->first) {
          throw std::runtime_error("node_name not match: " + iter_node_name->second + " vs. " +
                                   iter_config->first);
        }
      } else {
        static_config["node_name"] = iter_config->first;
      }
    }
    // config.clear();
    config[static_config["node_name"]] = static_config;
  }
  (*shared_dict)["config"] = config;
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

  CThreadSafeInterpreters::getInstance().append(this);

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

Interpreter::~Interpreter() {
  if (PyGILState_Check()) {
    py::gil_scoped_release gil_lock;  // https://github.com/pybind/pybind11/issues/1446
    backend_.reset();
  }
}

void Interpreter::forward(py::list py_inputs) {
  std::vector<dict> inputs;
  std::vector<dict> async_inputs;
  std::vector<dict> sync_inputs;
  std::set<int> sync_index;

  for (std::size_t i = 0; i < py::len(py_inputs); ++i) {
    IPIPE_ASSERT(py::isinstance<py::dict>(py_inputs[i]));
    inputs.push_back(py2dict(py_inputs[i]));
  }

  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto& item = inputs[i];
    auto iter = item->find(TASK_EVENT_KEY);
    if (iter != item->end()) {
      std::shared_ptr<SimpleEvents> input_event =
          any_cast<std::shared_ptr<SimpleEvents>>(iter->second);

      if (!input_event->valid())
        // input_event is invalid, so we cannot use it to store the exception;
        throw py::value_error("Invalid input event. Maybe it has been used before.");
      // add_const_callback need gil lock to handle py_input
      auto py_input = py_inputs[i];
      input_event->add_callback([item, py_input]() {
        py::gil_scoped_acquire gil_lock;
        dict2py(item, py_input);
      });
      async_inputs.push_back(item);
    } else {
      sync_inputs.push_back(item);
      sync_index.insert(i);
    }
  }
  {
    py::gil_scoped_release gil_lock;
    if (!async_inputs.empty()) forward(async_inputs);
    if (!sync_inputs.empty()) forward(sync_inputs);
  }

  for (const auto& i : sync_index) {
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
    // add_const_callback need gil lock to handle py_input
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
  if (!backend_) throw std::runtime_error("Interpreter uninitialized");
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

CThreadSafeInterpreters& CThreadSafeInterpreters::getInstance() {
  static CThreadSafeInterpreters inst;
  return inst;
}

void CThreadSafeInterpreters::append(Interpreter* inter) {
  std::lock_guard<std::mutex> lock(mutex_);
  interpreters_.push_back(inter);
}

std::vector<Interpreter*> CThreadSafeInterpreters::get() {
  std::lock_guard<std::mutex> lock(mutex_);
  return interpreters_;
}

std::once_flag Interpreter::once_flag_;

}  // namespace ipipe