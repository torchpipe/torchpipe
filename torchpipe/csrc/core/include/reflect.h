// Copyright 2010, The TOFT Authors.
// Author: CHEN Feng <chen3feng@gmail.com>
// BSD 3-Clause License

// Copyright (c) 2017, Tencent Inc.
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.

// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.

// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// modified from
// https://github.com/chen3feng/toft/blob/98ead831b1335034cb5991d8e7b1e6a8a7c2325d/base/class_registry/class_registry.h#L108

// learned from
// https://github.com/AngryHacker/articles/blob/master/src/c_plus_cplus/reflection_in_c%2B%2B_3.md

#ifndef __IPIPE_REFLECT_H__
#define __IPIPE_REFLECT_H__

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <string>
#include <vector>
#include "ipipe_common.hpp"
namespace {
std::vector<std::string> inline_str_split(std::string strtem, char a) {
  std::vector<std::string> strvec;
  if (strtem.empty()) return strvec;

  auto itor = std::remove(strtem.begin(), strtem.end(), ' ');
  strtem.erase(itor, strtem.end());

  std::string::size_type pos1, pos2;
  pos2 = strtem.find(a);
  pos1 = 0;
  while (std::string::npos != pos2) {
    strvec.push_back(strtem.substr(pos1, pos2 - pos1));

    pos1 = pos2 + 1;
    pos2 = strtem.find(a, pos1);
  }
  strvec.push_back(strtem.substr(pos1));
  for (auto iter_vec = strvec.begin(); iter_vec != strvec.end();) {
    if (iter_vec->empty())
      iter_vec = strvec.erase(iter_vec);
    else
      ++iter_vec;
  }
  return strvec;
}
}  // namespace

namespace ipipe {
namespace reflect {
IPIPE_EXPORT bool ipipe_load();
IPIPE_EXPORT void printlog_and_throw(std::string name);

template <typename ClassName>
class IPIPE_EXPORT ClassRegistryBase {
 public:
  ClassRegistryBase() { ipipe_load(); }
  using ObjectGetter = std::function<ClassName*()>;

  std::vector<std::string> IpipeGetAll() {
    std::vector<std::string> result;
    for (auto iter = getter_map_.begin(); iter != getter_map_.end(); ++iter) {
      result.push_back(iter->first);
    }
    return result;
  }

 private:
  typedef std::unordered_map<std::string, ObjectGetter> ClassMap;
  ClassMap getter_map_;
  mutable std::unordered_map<ClassName*, std::string> class_name_map_;

 public:
  void DoAddClass(const std::string& class_name, ObjectGetter getter) {
    auto keys = ::inline_str_split(class_name, ',');
    for (const auto& item : keys) DoAddSingleClass(item, getter);
  }

  void DoAddSingleClass(const std::string class_name, ObjectGetter getter) {
    typename ClassMap::iterator it = getter_map_.find(class_name);

    if (it != getter_map_.end()) {
      return;
    }
    getter_map_[class_name] = getter;
  }

  ClassName* DoGetObject(const std::string& class_name) const {
    typename ClassMap::const_iterator it = getter_map_.find(class_name);

    if (it == getter_map_.end()) {
      printlog_and_throw("The backend `" + class_name + "` is not registered.");

      return nullptr;
    }
    auto result = ((it->second))();
    if (result) {
      class_name_map_[result] = class_name;
    }
    return result;
  }

  std::string GetObjectName(ClassName* name) {
    auto iter = class_name_map_.find(name);
    if (iter != class_name_map_.end()) {
      return iter->second;
    } else {
      return "";
    }
  }

  ClassMap& IpipeGetMap(const std::string class_name) const { return getter_map_; }
};

template <typename RegistryType>
ClassRegistryBase<RegistryType>& ClassRegistryInstance() {
  static ClassRegistryBase<RegistryType> class_register;
  return class_register;
}

// All class can share the same creator as a function template
template <typename BaseClassName, typename SubClassName>
BaseClassName* ClassRegistry_NewObject() {
  return new SubClassName();
}

template <typename BaseClassName>
class IPIPE_EXPORT ClassRegisterer {
 public:
  ClassRegisterer(typename ClassRegistryBase<BaseClassName>::ObjectGetter getterraw,
                  std::string class_name, std::initializer_list<std::string> names) {
    ipipe_load();
    if (names.size() > 1) {
      for (auto iter = names.begin() + 1; iter != names.end(); ++iter)
        ClassRegistryInstance<BaseClassName>().DoAddClass(*iter, getterraw);
    } else {
      ClassRegistryInstance<BaseClassName>().DoAddClass(class_name, getterraw);
    }
  }

  ~ClassRegisterer() {}
};

}  // namespace reflect
}  // namespace ipipe

// class_name should not contain '>' '<' and ':'.
#define IPIPE_REGISTER(base_class_type, class_name, ...)                                 \
  static ipipe::reflect::ClassRegisterer<base_class_type> class_name##RegistryTag(       \
      ipipe::reflect::ClassRegistry_NewObject<base_class_type, class_name>, #class_name, \
      {#class_name, ##__VA_ARGS__});

#define IPIPE_CREATE(base_class_type, register_name) \
  ipipe::reflect::ClassRegistryInstance<base_class_type>().DoGetObject(register_name)

#define IPIPE_GET_REGISTER_NAME(base_class_type, obj_ptr) \
  ipipe::reflect::ClassRegistryInstance<base_class_type>().GetObjectName(obj_ptr)

#define IPIPE_ALL_NAMES(base_class_type) \
  ipipe::reflect::ClassRegistryInstance<base_class_type>().IpipeGetAll()

#endif