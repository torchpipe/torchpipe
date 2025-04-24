

#include <sstream>

#include "hami/core/dict.hpp"

#include <pybind11/pybind11.h>
// #include <pybind11/functional>
#include "hami/csrc/all2numpy.hpp"

#include <pybind11/stl.h>

#include "hami/core/queue.hpp"
#include "hami/csrc/converts.hpp"
#include "hami/csrc/dict.hpp"

#include "hami/csrc/py_register.hpp"
#include "hami/helper/base_logging.hpp"
#include "hami/helper/macro.h"
#include "hami/builtin/page_table.hpp"
#include "hami/core/event.hpp"

namespace hami {

namespace py = pybind11;
using namespace pybind11::literals;

PyDict::PyDict(const py::dict& data) {
  data_ = make_dict();
  for (const auto& item : data) {
    const std::string key = py::cast<std::string>(item.first);
    auto second = object2any(py::reinterpret_borrow<py::object>(item.second));
    if (second == std::nullopt) {
      throw py::type_error("hami.Any: The input type is unknown.");
    }
    data_->insert_or_assign(key, *second);
  }
}

PyDict::PyDict(dict data) : data_(data) {
  HAMI_ASSERT(data_ != nullptr, "The input data is nullptr.");
}

py::object PyDict::pop(
    const std::string& key,
    std::optional<std::string> default_value) {
  auto it = data_->find(key);
  if (it != data_->end()) {
    auto re = any2object(it->second);
    data_->erase(it);
    return re;
  } else if (default_value.has_value()) {
    return py::cast(default_value.value());
  }
  throw py::key_error("Key not found: " + key);
}

dict PyDict::py2dict(py::dict data) {
  dict result = make_dict();
  for (const auto& item : data) {
    const std::string key = py::cast<std::string>(item.first);
    auto second = object2any(item.second);
    if (second == std::nullopt) {
      throw py::type_error("hami.Any: The input type is unknown.");
    }
    result->insert_or_assign(key, *second);
  }
  return result;
}
void PyDict::dict2py(
    dict data,
    py::dict result,
    const std::unordered_set<std::string>& ignore_keys) {
  for (auto iter = data->begin(); iter != data->end(); ++iter) {
    if (ignore_keys.count(iter->first) != 0)
      continue;
    result[py::str(iter->first)] = any2object(iter->second);
  }
}

void PyDict::set(const std::string& key, const py::object& value) {
  auto data = object2any(value);
  if (data == std::nullopt)
    throw py::type_error("The input type is unknown by hami.Any.");
  data_->insert_or_assign(key, *data);
  // (*data)[key] = data;
}

std::shared_ptr<Event> PyDict::set_event() {
  auto ev = make_event();
  (*data_)[TASK_EVENT_KEY] = ev;
  return ev;
}

void PyDict::set(const std::string& key, const str::str_map& value) {
  data_->insert_or_assign(key, value);
}

py::object PyDict::get(const std::string& key) const {
  auto it = data_->find(key);
  if (it != data_->end()) {
    return any2object(it->second);
  }
  throw py::key_error("Key not found: " + key);
}

void init_dict(py::module_& m) {
  py::class_<PyDict> hami_dict(m, "Dict");

  hami_dict.doc() =
      "hami.Dict provides an object wrapper for the "
      "hami::dict class, which is essentially a wrapper around "
      "std::shared_ptr<std::unordered_map<std::string, std::any>>.";
  hami_dict.def(py::init<>())
      .def(
          py::init<const py::dict&>(),
          "Construct a dictionary from a Python dict")
      // .def("set", &PyDict::set, "Set a value in the dictionary", "key"_a,
      // "value"_a)
      .def("get", &PyDict::get, "Get a value from the dictionary", "key"_a)
      .def(
          "contains",
          &PyDict::contains,
          "Check if the dictionary contains a key",
          "key"_a)
      .def(
          "remove",
          &PyDict::remove,
          "Remove a key from the dictionary",
          "key"_a)
      .def("clear", &PyDict::clear, "Clear the dictionary")
      .def("set_event", &PyDict::set_event, "set and return the event")
      .def(
          "pop",
          &PyDict::pop,
          py::arg("key"),
          py::arg("default") = py::none(),
          "Remove specified key and return the corresponding value.\n"
          "If key is not found, default is returned if given, otherwise "
          "KeyError is raised.")
      .def(
          "__setitem__",
          py::overload_cast<const std::string&, const py::object&>(
              &PyDict::set),
          "Set a value in the dictionary",
          "key"_a,
          "value"_a)
      .def(
          "__setitem__",
          py::overload_cast<
              const std::string&,
              const std::unordered_map<std::string, std::string>&>(
              &PyDict::set),
          "Set a nested dictionary value in the dictionary",
          "key"_a,
          "value"_a)
      .def(
          "__getitem__",
          &PyDict::get,
          py::return_value_policy::reference_internal,
          "Get a value from the dictionary",
          py::arg("key"))
      .def(
          "update",
          py::overload_cast<const PyDict&>(&PyDict::update),
          "Update the dictionary with another PyDict",
          "other"_a)
      .def(
          "update",
          py::overload_cast<const str::str_map&>(&PyDict::update),
          "Update the dictionary with a str_map",
          "other"_a)
      .def(
          "__contains__",
          &PyDict::contains,
          "Check if the dictionary contains a key",
          "key"_a)
      .def(
          "__delitem__",
          &PyDict::remove,
          "Remove a key from the dictionary",
          "key"_a)
      .def(
          "keys",
          [](const PyDict& d) {
            return py::make_key_iterator(d.data().begin(), d.data().end());
          },
          py::return_value_policy::reference_internal)
      .def(
          "__iter__",
          [](const PyDict& d) {
            return py::make_iterator(d.data().begin(), d.data().end());
          },
          py::keep_alive<0, 1>())
      .def(
          "__len__", &PyDict::size, "Get the number of items in the dictionary")
      .def(
          "__repr__",
          [](const PyDict& d) {
            std::ostringstream repr_stm;
            repr_stm << "{";
            for (const auto& [key, value] : d.data()) {
              auto re = any2object(value);
              repr_stm << key << ": " << pybind11::repr(re).cast<std::string>()
                       << ", "; // re.attr("__repr__")()
            }
            std::string repr = repr_stm.str();
            if (repr.size() > 1)
              repr.pop_back(), repr.pop_back();
            repr += "}";
            return repr;
          })
      .def("__repr__2", [](const PyDict& d) {
        std::ostringstream repr_stm;
        repr_stm << "{";
        for (const auto& [key, value] : d.data()) {
          // auto re = any2object(value);

          repr_stm << key << ": <no_repr>, ";
        }
        std::string repr = repr_stm.str();
        if (repr.size() > 2)
          repr.pop_back(), repr.pop_back();
        repr += "}";
        return repr;
      });

  py::class_<TypedDict, std::shared_ptr<TypedDict>>(m, "TypedDict")
      .def(py::init<>())
      .def(py::init<std::unordered_map<std::string, TypedDict::BaseType>>())
      .def(py::init([](py::dict d) {
        auto self = std::make_shared<TypedDict>();
        for (auto item : d) {
          std::string key = py::cast<std::string>(item.first);

          try {
            self->data.emplace(key, py::cast<TypedDict::BaseType>(item.second));
          } catch (const py::cast_error& e) {
            throw py::value_error(
                "Value for key '" + key + "' has invalid type: " + e.what());
          }
        }
        return self;
      }))
      .def(
          "__getitem__",
          [](const TypedDict& self, const std::string& key) {
            if (self.data.find(key) == self.data.end()) {
              throw py::key_error("Key '" + key + "' not found");
            }
            return self.data.at(key);
          })
      .def(
          "__setitem__",
          [](TypedDict& self,
             const std::string& key,
             const TypedDict::BaseType& value) { self.data[key] = value; })
      .def(
          "__contains__",
          [](const TypedDict& self, const std::string& key) {
            return self.data.find(key) != self.data.end();
          })
      .def("__len__", [](const TypedDict& self) { return self.data.size(); })
      .def_property_readonly(
          "data", [](const TypedDict& self) { return self.data; })
      .def(
          "__repr__",
          [](const TypedDict& self) {
            std::ostringstream oss;
            oss << "TypedDict({";
            for (const auto& [key, value] : self.data) {
              oss << "'" << key << "': ";

              // 自定义输出处理器
              auto value_printer = [&oss](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
                  oss << "[";
                  for (size_t i = 0; i < arg.size(); ++i) {
                    oss << arg[i];
                    if (i != arg.size() - 1)
                      oss << ", ";
                  }
                  oss << "]";
                } else if constexpr (std::is_same_v<
                                         T,
                                         std::vector<std::string>>) {
                  oss << "[";
                  for (size_t i = 0; i < arg.size(); ++i) {
                    oss << "'" << arg[i] << "'";
                    if (i != arg.size() - 1)
                      oss << ", ";
                  }
                  oss << "]";
                } else {
                  oss << arg;
                }
              };

              std::visit(value_printer, value);
              oss << ", ";
            }
            if (!self.data.empty())
              oss.seekp(-2, oss.cur);
            oss << "})";
            return oss.str();
          })
      .attr("type_hash") = typeid(std::shared_ptr<TypedDict>).hash_code();

  // Register base exception
  py::register_exception<hami::queue::QueueException>(m, "QueueError");

  // Try to map our exceptions to Python's queue module exceptions
  try {
    // Import Python's queue module to get standard exception types
    py::module queue_module = py::module::import("queue");
    py::object py_empty = queue_module.attr("Empty");
    py::object py_full = queue_module.attr("Full");

    // Register exceptions with correct inheritance relationship
    py::exception<hami::queue::QueueEmptyException>(m, "Empty", py_empty.ptr());
    py::exception<hami::queue::QueueFullException>(m, "Full", py_full.ptr());
  } catch (...) {
    // If import fails, register with standard exceptions
    SPDLOG_ERROR(
        "Failed to import Python's queue module. register with standard "
        "exceptions.");
    py::register_exception<hami::queue::QueueEmptyException>(m, "Empty");
    py::register_exception<hami::queue::QueueFullException>(m, "Full");
  }

  m.def("print", [](const std::string& info) {
    py::gil_scoped_release release;

    hami::default_logger_raw()->info(info);
  });

  m.def(
      "default_queue",
      &default_queue,
      py::arg("tag") = std::string(""),
      pybind11::return_value_policy::reference);
  m.def(
      "default_src_queue",
      &default_src_queue,
      pybind11::return_value_policy::reference);
  m.def(
      "default_output_queue",
      &default_output_queue,
      pybind11::return_value_policy::reference);

  // Bind PageInfo structure first
  py::class_<PageTable::PageInfo>(m, "PageInfo")
      .def(py::init<>())
      // .def_readwrite("kv_page_indices",
      // &PageTable::PageInfo::kv_page_indices)
      .def_property(
          "kv_page_indices",
          [](const PageTable::PageInfo& self) {
            auto& vec = self.kv_page_indices;
            // 创建非拷贝数组视图
            return py::array_t<int>(
                {vec.size()}, // shape
                {sizeof(int)}, // stride
                vec.data(), // 原始指针
                py::cast(self) // 保持父对象存活
            );
          },
          [](PageTable::PageInfo& self, py::array_t<int> arr) {
            py::buffer_info buf = arr.request();
            if (buf.ndim != 1)
              throw std::runtime_error("Only 1D arrays accepted");

            self.kv_page_indices.resize(buf.shape[0]);
            std::memcpy(
                self.kv_page_indices.data(),
                buf.ptr,
                sizeof(int) * buf.shape[0]);
          })
      .def_readwrite(
          "kv_last_page_len", &PageTable::PageInfo::kv_last_page_len);

  // Main PageTable class binding
  py::class_<PageTable, std::shared_ptr<PageTable>>(m, "PageTable")
      .def(py::init<>())
      .def(
          py::init<size_t, size_t, size_t>(),
          py::arg("max_num_req"),
          py::arg("max_num_page"),
          py::arg("page_size"))
      .def(
          "init",
          [](PageTable& self,
             size_t max_num_req,
             size_t max_num_page,
             size_t page_size) -> PageTable& {
            self.init(max_num_req, max_num_page, page_size);
            return self; // 返回 self 实现链式调用
          },
          py::arg("max_num_req"),
          py::arg("max_num_page"),
          py::arg("page_size"),
          py::return_value_policy::reference_internal)
      .def("alloc", &PageTable::alloc, py::arg("id"), py::arg("num_tok"))
      .def("reset", &PageTable::reset, py::arg("id"), py::arg("num_tok"))
      .def("extend", &PageTable::extend, py::arg("id"))
      .def("free", &PageTable::free, py::arg("id"))
      .def(
          "page_table",
          [](PageTable& self, const std::vector<id_type>& id) {
            std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> re;
            {
              py::gil_scoped_release release;
              re = self.page_table(id);
            }

            return py::make_tuple(
                to_numpy(std::move(std::get<0>(re))),
                to_numpy(std::move(std::get<1>(re))),
                to_numpy(std::move(std::get<2>(re))));
          },
          py::arg("id"))
      .def(
          "add_more_page",
          &PageTable::add_more_page,
          py::arg("num_added_slots"))
      .def(
          "available_pages",
          &PageTable::available_pages,
          pybind11::call_guard<pybind11::gil_scoped_release>())
      .def(
          "get_num_tok",
          &PageTable::get_num_tok,
          pybind11::call_guard<pybind11::gil_scoped_release>())
      .def(
          "get_prefill_num_req_toks",
          &PageTable::get_prefill_num_req_toks,
          py::arg("id"),
          pybind11::call_guard<pybind11::gil_scoped_release>())
      .def(
          "available_ids",
          &PageTable::available_ids,
          pybind11::call_guard<pybind11::gil_scoped_release>())
      // .def("pop_activated", &PageTable::pop_activated)
      .def(
          "pop_activated",
          [](PageTable& self) {
            auto result = self.pop_activated();
            return py::make_tuple(
                result.first, //  std::vector
                to_numpy(std::move(result.second)) // 转换为 NumPy 数组
            );
          })
      .def(
          "page_info",
          &PageTable::page_info,
          py::arg("id"),
          py::return_value_policy::reference_internal);

  // Default page table factory function
  m.def(
      "default_page_table",
      [](const std::string& tag) -> std::shared_ptr<PageTable> {
        // Assuming singleton-like management, prevent deletion
        return std::shared_ptr<PageTable>(
            &default_page_table(tag), [](PageTable*) { /* no-op deleter */ });
      },
      py::arg("tag") = "",
      py::return_value_policy::automatic);

  py::class_<Queue, std::shared_ptr<Queue>> queue_class(m, "Queue");
  py::enum_<typename Queue::Status>(queue_class, "Status")
      .value("RUNNING", Queue::Status::RUNNING)
      .value("ERROR", Queue::Status::ERROR)
      .value("PAUSED", Queue::Status::PAUSED)
      .value("CANCELED", Queue::Status::CANCELED)
      .value("EOS", Queue::Status::EOS)
      .export_values();
  queue_class.def(py::init<>())
      .def(
          "put",
          [](Queue& self, PyDict item) {
            auto data = item.to_dict();
            py::gil_scoped_release release;
            self.put(data);
          },
          py::arg("item"))
      .def(
          "get",
          [](Queue& self, bool block, std::optional<double> timeout) {
            dict result;

            {
              py::gil_scoped_release release;
              result = self.get(block, timeout);
            }
            return PyDict(result);
          },
          py::arg("block") = true,
          py::arg("timeout") = std::nullopt)
      .def(
          "wait_until_at_most",
          [](Queue& self, size_t max_size, double timeout_sec) {
            return self.wait_until_at_most(
                max_size, std::chrono::duration<double>(timeout_sec));
          },
          py::arg("max_size"),
          py::arg("timeout_sec"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "wait_until_at_least",
          [](Queue& self, size_t min_size, double timeout_sec) {
            return self.wait_until_at_least(
                min_size, std::chrono::duration<double>(timeout_sec));
          },
          py::arg("min_size"),
          py::arg("timeout_sec"),
          py::call_guard<py::gil_scoped_release>())
      .def("size", &Queue::size)
      .def("empty", &Queue::empty)
      .def("status", &Queue::status)
      .def("join", &Queue::join)
      .def_static("type_hash", []() {
        // SPDLOG_INFO("type_hash : {}", typeid(Queue*).name());
        return typeid(Queue).hash_code();
      });

  reg::register_any_ptr_object_hash_converter<Queue>();
}

} // namespace hami
