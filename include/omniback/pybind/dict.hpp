#pragma once
#include <pybind11/pybind11.h>
#include "omniback/core/dict.hpp"

namespace omniback {
class Event;
namespace py = pybind11;
class PyDict {
 public:
  PyDict() : data_(make_dict()) {}
  PyDict(dict data);
  PyDict(const py::dict& data);

  void set(const std::string& key, const py::object& value);
  void set(const std::string& key, const str::str_map& value);
  std::shared_ptr<Event> set_event();

  py::object get(const std::string& key) const;

  bool contains(const std::string& key) const {
    return data_->find(key) != data_->end();
  }

  void clear() {
    data_->clear();
  }
  void remove(const std::string& key) {
    data_->erase(key);
  }

  py::object pop(const std::string& key, py::object default_val = py::none());

  py::object pop_str(
      const std::string& key,
      std::optional<std::string> default_value = std::nullopt);

  const std::unordered_map<std::string, omniback::any>& data() const {
    return *data_;
  }
  std::unordered_map<std::string, omniback::any>& data() {
    return *data_;
  }

  dict to_dict() const {
    return data_;
  }
  static dict py2dict(py::dict data);
  static void dict2py(
      dict data,
      py::dict result,
      const std::unordered_set<std::string>& ignore_keys);

  void update(const PyDict& other) {
    for (const auto& item : *other.data_) {
      data_->insert_or_assign(item.first, item.second);
    }
  }
  void update(const str::str_map& other) {
    for (const auto& item : other) {
      data_->insert_or_assign(item.first, item.second);
    }
  }
  size_t size() const {
    return data_->size();
  }
  bool empty() const {
    return data_->empty();
  }

 private:
  dict data_;
};

struct PyStrMap {
  std::unordered_map<std::string, std::string> data;
};

} // namespace omniback