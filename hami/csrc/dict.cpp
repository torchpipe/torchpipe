

#include <sstream>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "hami/core/dict.hpp"
#include "hami/csrc/dict.hpp"
#include "hami/csrc/converts.hpp"
#include "hami/helper/macro.h"
#include "hami/core/queue.hpp"
#include "hami/helper/base_logging.hpp"

namespace hami {

namespace py = pybind11;
using namespace pybind11::literals;

PyDict::PyDict(const py::dict& data) {
    data_ = make_dict();
    for (const auto& item : data) {
        const std::string key = py::cast<std::string>(item.first);
        auto second = object2any(item.second);
        if (second == std::nullopt) {
            throw py::type_error("hami.Any: The input type is unknown.");
        }
        data_->insert_or_assign(key, *second);
    }
}

PyDict::PyDict(dict data) : data_(data) {
    HAMI_ASSERT(data_ != nullptr, "The input data is nullptr.");
}

py::object PyDict::pop(const std::string& key,
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
void PyDict::dict2py(dict data, py::dict result,
                     const std::unordered_set<std::string>& ignore_keys) {
    for (auto iter = data->begin(); iter != data->end(); ++iter) {
        if (ignore_keys.count(iter->first) != 0) continue;
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
        .def(py::init<const py::dict&>(),
             "Construct a dictionary from a Python dict")
        // .def("set", &PyDict::set, "Set a value in the dictionary", "key"_a,
        // "value"_a)
        .def("get", &PyDict::get, "Get a value from the dictionary", "key"_a)
        .def("contains", &PyDict::contains,
             "Check if the dictionary contains a key", "key"_a)
        .def("remove", &PyDict::remove, "Remove a key from the dictionary",
             "key"_a)
        .def("clear", &PyDict::clear, "Clear the dictionary")
        .def("pop", &PyDict::pop, py::arg("key"),
             py::arg("default") = py::none(),
             "Remove specified key and return the corresponding value.\n"
             "If key is not found, default is returned if given, otherwise "
             "KeyError is raised.")
        .def("__setitem__",
             py::overload_cast<const std::string&, const py::object&>(
                 &PyDict::set),
             "Set a value in the dictionary", "key"_a, "value"_a)
        .def("__setitem__",
             py::overload_cast<
                 const std::string&,
                 const std::unordered_map<std::string, std::string>&>(
                 &PyDict::set),
             "Set a nested dictionary value in the dictionary", "key"_a,
             "value"_a)
        .def("__getitem__", &PyDict::get, "Get a value from the dictionary",
             "key"_a)
        .def("update", py::overload_cast<const PyDict&>(&PyDict::update),
             "Update the dictionary with another PyDict", "other"_a)
        .def("update", py::overload_cast<const str::str_map&>(&PyDict::update),
             "Update the dictionary with a str_map", "other"_a)
        .def("__contains__", &PyDict::contains,
             "Check if the dictionary contains a key", "key"_a)
        .def("__delitem__", &PyDict::remove, "Remove a key from the dictionary",
             "key"_a)
        .def("__len__", &PyDict::size,
             "Get the number of items in the dictionary")
        .def("__repr__", [](const PyDict& d) {
            std::ostringstream repr_stm;
            repr_stm << "{";
            for (const auto& [key, value] : d.data()) {
                auto re = any2object(value);
                repr_stm << key << ": " << re.attr("__repr__")() << ", ";
            }
            std::string repr = repr_stm.str();
            if (repr.size() > 1) repr.pop_back(), repr.pop_back();
            repr += "}";
            return repr;
        });
    // Register base exception
    py::register_exception<hami::queue::QueueException>(m, "QueueError");

    // Try to map our exceptions to Python's queue module exceptions
    try {
        // Import Python's queue module to get standard exception types
        py::module queue_module = py::module::import("queue");
        py::object py_empty = queue_module.attr("Empty");
        py::object py_full = queue_module.attr("Full");

        // Register exceptions with correct inheritance relationship
        py::exception<hami::queue::QueueEmptyException>(m, "Empty",
                                                        py_empty.ptr());
        py::exception<hami::queue::QueueFullException>(m, "Full",
                                                       py_full.ptr());
    } catch (...) {
        // If import fails, register with standard exceptions
        SPDLOG_ERROR(
            "Failed to import Python's queue module. register with standard "
            "exceptions.");
        py::register_exception<hami::queue::QueueEmptyException>(m, "Empty");
        py::register_exception<hami::queue::QueueFullException>(m, "Full");
    }

    m.def("default_queue", &default_queue, py::arg("tag") = std::string(""),
          pybind11::return_value_policy::reference);
    m.def("default_src_queue", &default_src_queue,
          pybind11::return_value_policy::reference);
    m.def("default_output_queue", &default_output_queue,
          pybind11::return_value_policy::reference);

    py::class_<Queue>(m, "Queue")
        .def(py::init<>())
        .def(
            "put",
            [](Queue& self, PyDict item, size_t size) {
                auto data = item.to_dict();
                py::gil_scoped_release release;
                self.put(data, size);
            },
            py::arg("item"), py::arg("size") = 1)
        .def(
            "get",
            [](Queue& self, bool block, std::optional<double> timeout) {
                if (block) {
                    std::pair<dict, size_t> result;
                    {
                        py::gil_scoped_release release;
                        result = self.get(block, timeout);
                    }
                    return std::pair<PyDict, size_t>(PyDict(result.first),
                                                     result.second);
                } else {
                    auto re = self.get(block, timeout);
                    return std::pair<PyDict, size_t>(PyDict(re.first),
                                                     re.second);
                }
            },
            py::arg("block") = true, py::arg("timeout") = std::nullopt)
        .def("size", &Queue::size)
        .def("empty", &Queue::empty);
}

}  // namespace hami
