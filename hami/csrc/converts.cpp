#include "hami/csrc/converts.hpp"

#include <pybind11/stl.h>

#include "hami/core/dict.hpp"
#include "hami/helper/base_logging.hpp"
#include "hami/csrc/py_register.hpp"
#include "hami/helper/string.hpp"
#include "hami/csrc/register.hpp"

PYBIND11_MAKE_OPAQUE(hami::str::str_map);

namespace hami
{
    namespace py = pybind11;

    std::optional<any> object2any_base_type(pybind11::handle data)
    {
        if (py::isinstance<py::str>(data))
        {
            return py::cast<std::string>(data);
        }
        else if (py::isinstance<py::int_>(data))
        {
            return py::cast<int>(data);
        }
        else if (py::isinstance<py::float_>(data))
        {
            return py::cast<float>(data);
        }
        else if (py::isinstance<py::bytes>(data))
        {
            return py::cast<std::string>(data);
        }
        else if (py::isinstance<TypedDict>(data))
        {
            return py::cast<std::shared_ptr<TypedDict>>(data);
        }
        else if (py::isinstance<str::str_map>(data))
        {
            return py::cast<str::str_map>(data);
        }
        return std::nullopt;
    }

    std::optional<any> object2any_list_base_type(const pybind11::object &inner_data,
                                                 const pybind11::object &data)
    {
        if (py::isinstance<py::str>(inner_data))
        {
            return convert_list<std::string>(data);
        }
        else if (py::isinstance<py::int_>(inner_data))
        {
            return convert_list<int>(data);
        }
        else if (py::isinstance<py::float_>(inner_data))
        {
            return convert_list<float>(data);
        }
        else if (py::isinstance<py::bytes>(inner_data))
        {
            return convert_list<std::string>(data);
        }
        else if (py::isinstance<TypedDict>(inner_data))
        {
            return convert_list<std::shared_ptr<TypedDict>>(data);
        }
        return std::nullopt;
    }

    std::optional<any> object2any_dict_base_type(const pybind11::handle &inner_data,
                                                 const pybind11::dict &data)
    {
        if (py::isinstance<py::str>(inner_data))
        {
            return convert_dict<std::string, std::string>(data);
        }
        else if (py::isinstance<py::int_>(inner_data))
        {
            return convert_dict<std::string, int>(data);
        }
        else if (py::isinstance<py::float_>(inner_data))
        {
            return convert_dict<std::string, float>(data);
        }
        else if (py::isinstance<py::bytes>(inner_data))
        {
            return convert_dict<std::string, std::string>(data);
        }
        else if (py::isinstance<py::bytes>(inner_data))
        {
            return convert_dict<std::string, std::string>(data);
        }
        else if (py::isinstance<py::dict>(inner_data))
        {
            py::dict inindata = py::cast<py::dict>(inner_data);
            if (0 == py::len(inindata))
            {
                SPDLOG_DEBUG("dict is empty.");
                throw std::runtime_error(EMPTY_DICT_NOT_CONVERTABLE);
            }
            if (!py::isinstance<py::str>(inindata.begin()->first))
                return std::nullopt;

            str::mapmap re;
            for (const auto &item : data)
            {
                auto item_dict = py::cast<py::dict>(item.second);
                const std::string key = item.first.cast<std::string>();
                if (py::len(item_dict) == 0)
                {
                    re[key] = str::str_map();
                }
                else
                {
                    auto inner = str::str_map();

                    for (const auto &inner_item : item_dict)
                    {
                        const std::string inner_key =
                            inner_item.first.cast<std::string>();
                        std::string inner_value =
                            py::cast<std::string>(inner_item.second);
                        inner[inner_key] = inner_value;
                    }
                    re[key] = inner;
                }
            }
            return re;
        }
        return std::nullopt;
    }

    std::optional<any> object2any_set_base_type(const pybind11::object &inner_data,
                                                const pybind11::object &data)
    {
        if (py::isinstance<py::str>(inner_data))
        {
            return py::cast<std::vector<std::string>>(data);
        }
        else if (py::isinstance<py::int_>(inner_data))
        {
            return py::cast<std::vector<int>>(data);
        }
        else if (py::isinstance<py::float_>(inner_data))
        {
            return py::cast<std::vector<float>>(data);
        }
        else if (py::isinstance<py::bytes>(inner_data))
        {
            return py::cast<std::vector<std::string>>(data);
        }
        else if (py::isinstance<TypedDict>(inner_data))
        {
            return py::cast<std::vector<std::shared_ptr<TypedDict>>>(data);
        }
        return std::nullopt;
    }

    std::optional<any> object2any(const py::handle &obj)
    {
        auto re = convert_py2any(obj);
        if (re)
            return re;

        if (py::isinstance<py::list>(obj))
        {
            py::list inner_list = py::cast<py::list>(obj);

            if (0 == py::len(inner_list))
            {
                SPDLOG_DEBUG("inner list is empty.");
                throw std::runtime_error("inner list is empty.");
            }

            return object2any_list_base_type(inner_list[0], inner_list);
        }
        if (py::isinstance<py::dict>(obj))
        {
            py::dict data = py::cast<py::dict>(obj);
            if (py::len(data) == 0)
            {
                SPDLOG_DEBUG("Dictionary is empty.");
                throw std::runtime_error(EMPTY_DICT_NOT_CONVERTABLE);
            }
            if (!py::isinstance<py::str>(data.begin()->first))
                return std::nullopt;
            return object2any_dict_base_type(data.begin()->second, data);
        }
        else if (py::isinstance<py::tuple>(obj))
        {
            throw py::type_error(
                "The input data(tuple) is not supported by hami.Any. Use List "
                "instead.");
        }
        else
        {
            auto re = object2any_base_type(obj);
            return re;
        }
        return std::nullopt;
    }
    py::object any2object(const any &input)
    {
        return hami::reg::any2object_from_hash_register(input);
    }

    // class PyConverterRegistry {
    //  public:
    //   static PyConverterRegistry& instance() {
    //     static PyConverterRegistry registry;
    //     return registry;
    //   }

    //   void add_converter(std::function<std::optional<any>(const
    //   pybind11::handle&)> converter) {
    //     converters_.push_back(std::move(converter));
    //   }

    //   std::optional<any> try_convert(const pybind11::handle& obj) const {
    //     for (const auto& converter : converters_) {
    //       if (auto result = converter(obj)) {
    //         return result;
    //       }
    //     }
    //     return std::nullopt;
    //   }

    //  private:
    //   PyConverterRegistry() = default;
    //   std::vector<std::function<std::optional<any>(const pybind11::handle&)>>
    //   converters_;
    // };

    // void register_py2cpp(std::function<std::optional<any>(const
    // pybind11::handle&)> converter) {
    //   PyConverterRegistry::instance().add_converter(std::move(converter));
    // }
    // void register_cpp2py(std::function<std::optional<pybind11::object>(const
    // hami::any&)> converter)
    // {} const std::vector<std::function<std::optional<pybind11::object>(const
    // hami::any&)>>& get_cpp2py_registers() {} const
    // std::vector<std::function<std::optional<hami::any>(const
    // pybind11::handle&)>>& get_py2cpp_registers() {}
} // namespace hami