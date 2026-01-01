#pragma once

#define DEFINE_CONVERSION_FUNCTIONS_TO_PY(type, name)                                              \
  def("as_" #name, [](const any& self) { return any_cast<type>(self); })                           \
      .def("as_list_of_" #name, [](const any& self) { return any_cast<std::vector<type>>(self); }) \
      .def("as_set_of_" #name,                                                                     \
           [](const any& self) { return any_cast<std::unordered_set<type>>(self); })               \
      .def("as_dict_of_" #name,                                                                    \
           [](const any& self) { return any_cast<std::unordered_map<string, type>>(self); })

#define DEFINE_CONVERSION_FUNCTIONS_TO_PY(type, name)                                              \
  def("as_" #name, [](const any& self) { return any_cast<type>(self); })                           \
      .def("as_list_of_" #name, [](const any& self) { return any_cast<std::vector<type>>(self); }) \
      .def("as_set_of_" #name,                                                                     \
           [](const any& self) { return any_cast<std::unordered_set<type>>(self); })               \
      .def("as_dict_of_" #name,                                                                    \
           [](const any& self) { return any_cast<std::unordered_map<string, type>>(self); })
