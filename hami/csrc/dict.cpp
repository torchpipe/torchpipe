

#include <sstream>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "hami/core/dict.hpp"
#include "hami/csrc/dict.hpp"
#include "hami/csrc/converts.hpp"
#include "hami/helper/macro.h"

namespace hami {

namespace py = pybind11;
using namespace pybind11::literals;

PyDict::PyDict(const py::dict& data) {
    data_ = make_dict();
    for (const auto& item : data) {
        const std::string key = py::cast<std::string>(item.first);
        auto second = object2any(item.second);
        if (second == std::nullopt) {
            throw py::type_error("hami.any: The input type is unknown.");
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
            throw py::type_error("hami.any: The input type is unknown.");
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
        throw py::type_error("The input type is unknown by hami.any.");
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
    py::class_<PyDict> hami_dict(m, "dict");

    hami_dict.doc() =
        "hami.dict provides an object wrapper for the "
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
}

}  // namespace hami
