// Copyright 2010, The TOFT Authors.
// Copyright: CHEN Feng <chen3feng@gmail.com>
// BSD 3-Clause License

// Copyright 2021-2025, Netease Inc.
// All rights reserved.

// modified from
// https://github.com/chen3feng/toft/blob/98ead831b1335034cb5991d8e7b1e6a8a7c2325d/base/class_registry/class_registry.h#L108

// learned from
// https://github.com/AngryHacker/articles/blob/master/src/c_plus_cplus/reflection_in_c%2B%2B_3.md

#ifndef __HAMI_REFLECT_H__
#define __HAMI_REFLECT_H__

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <map>
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <optional>

#include "hami/helper/macro.h"

namespace hami {
HAMI_EXPORT bool hami_load();
HAMI_EXPORT void printlog_and_throw(std::string name);
HAMI_EXPORT void printlog(std::string name);
HAMI_EXPORT std::vector<std::string> strict_str_split(std::string strtem, char a);
HAMI_EXPORT std::vector<std::array<std::string, 2>> multi_str_split(std::string strtem,
                                                                    char inner_sp, char outer);
HAMI_EXPORT void print_check_distance(std::string strtem, const std::vector<std::string>& targets);

template <typename ClassName>
class HAMI_EXPORT ClassRegistryBase {
 public:
  ClassRegistryBase() { hami_load(); }
  using ObjectGetter = std::function<ClassName*()>;

  std::vector<std::string> HamiGetAll() {
    std::vector<std::string> result;
    for (auto iter = getter_map_.begin(); iter != getter_map_.end(); ++iter) {
      result.push_back(iter->first);
    }
    return result;
  }

 private:
  typedef std::map<std::string, ObjectGetter> ClassMap;
  ClassMap getter_map_;
  std::mutex getter_map_mutex_;
  std::unordered_map<ClassName*, std::string> class_name_map_;
  std::unordered_map<std::string, ClassName*> reverse_class_name_map_;
  std::unordered_map<std::string, std::shared_ptr<ClassName>> reverse_class_name_map_owner_;

  std::mutex class_name_map_mutex_;

 public:
  void DoAddClass(const std::string& class_name, ObjectGetter getter) {
    auto keys = strict_str_split(class_name, ',');
    for (const auto& item : keys) DoAddSingleClass(item, getter);
  }

  void DoAddSingleClass(const std::string& class_name, ObjectGetter getter) {
    std::lock_guard<std::mutex> guard(getter_map_mutex_);
    typename ClassMap::iterator it = getter_map_.find(class_name);

    if (it != getter_map_.end()) {
      return;
    }
    getter_map_[class_name] = getter;
    // printlog("Register " + class_name);
  }

  ClassName* DoCreateObject(const std::string& class_name, const std::string& aspect_name = "") {
    std::unique_lock<std::mutex> guard(getter_map_mutex_);
    typename ClassMap::const_iterator it = getter_map_.find(class_name);

    if (it == getter_map_.end()) {
      std::string registered_cls;
      for (auto iter = getter_map_.begin(); iter != getter_map_.end(); ++iter) {
        registered_cls += iter->first + " ";
      }
      printlog(class_name + ": not registered. Registered: " + registered_cls);

      return nullptr;
    }
    auto result = ((it->second))();

    guard.unlock();
    if (result) {
      std::lock_guard<std::mutex> guard(class_name_map_mutex_);
      class_name_map_[result] = class_name;
      if (!aspect_name.empty()) {
        // printlog("Named Backend-Object: " + aspect_name + ". Class: " + class_name);
        reverse_class_name_map_[aspect_name] = result;
      }
    }
    return result;
  }

  // Why use shared_ptr? Because it facilitates easier interaction with Python.
  void DoRegisterObject(const std::string& aspect_name, const std::shared_ptr<ClassName>& backend) {
    {
      std::lock_guard<std::mutex> guard(class_name_map_mutex_);
      {
        printlog("Register Named Instance `" + aspect_name + std::string("` in address ") +
                 std::to_string((long long)&reverse_class_name_map_));
        reverse_class_name_map_[aspect_name] = backend.get();
        reverse_class_name_map_owner_[aspect_name] = backend;
      }
    }
  }

  void DoRegisterObject(const std::string& aspect_name, ClassName* backend) {
    {
      std::lock_guard<std::mutex> guard(class_name_map_mutex_);
      {
        printlog("Register Named Instance(wo/ ownership): " + aspect_name);
        if (reverse_class_name_map_.find(aspect_name) != reverse_class_name_map_.end()) {
          throw std::runtime_error("Duplicated backend name: " + aspect_name);
        }
        reverse_class_name_map_[aspect_name] = backend;
      }
    }
  }

  void DoUnRegisterObject(const std::string& aspect_name) {
    {
      std::lock_guard<std::mutex> guard(class_name_map_mutex_);
      {
        printlog("UnRegister Named Instance(w/ ownership): " + aspect_name);
        reverse_class_name_map_.erase(aspect_name);
        reverse_class_name_map_owner_.erase(aspect_name);
      }
    }
  }

  void DoCleanUp() {
    {
      std::lock_guard<std::mutex> guard(class_name_map_mutex_);
      {
        for (const auto& item : reverse_class_name_map_owner_) {
          reverse_class_name_map_.erase(item.first);
        }
        reverse_class_name_map_owner_.clear();
      }
    }
  }

  ClassName* DoGetObject(const std::string& aspect_name) {
    {
      std::lock_guard<std::mutex> guard(class_name_map_mutex_);
      auto iter = reverse_class_name_map_.find(aspect_name);
      if (iter != reverse_class_name_map_.end()) {
        printlog("Get Named Backend-Object: " + aspect_name);
        return iter->second;
      } else {
        printlog("Get Named Instance: " + aspect_name + " failed." + std::string(" address: ") +
                 std::to_string((long long)&reverse_class_name_map_));
        std::vector<std::string> keys;
        for (const auto& item : reverse_class_name_map_) {  // NOLINT
          keys.push_back(item.first);
        }
        print_check_distance(aspect_name, keys);
        // printlog("Get Named Backend-Object: " + aspect_name + " failed.");
        return nullptr;
      }
    }
  }

  std::optional<std::string> GetObjectName(ClassName* name) {
    std::lock_guard<std::mutex> guard(class_name_map_mutex_);
    auto iter = class_name_map_.find(name);
    if (iter != class_name_map_.end()) {
      return iter->second;
    } else {
      printlog("GetObjectName: not found . The class may not be created by reflection.");
      // return "";
    }
    return std::nullopt;
  }

  // std::optional<std::string> GetObjectName(ClassName* name) {
  //   std::lock_guard<std::mutex> guard(class_name_map_mutex_);
  //   auto iter = class_name_map_.find(name);
  //   if (iter != class_name_map_.end()) {
  //     return iter->second;
  //   }
  //   return std::nullopt;
  // }

  // ClassMap& HamiGetMap(const std::string class_name) const { return getter_map_; }
};

// template <typename RegistryType>
// ClassRegistryBase<RegistryType>& ClassRegistryInstance() {
//   static ClassRegistryBase<RegistryType> class_register;
//   return class_register;
// }

template <typename RegistryType>
HAMI_EXPORT ClassRegistryBase<RegistryType>& ClassRegistryInstance();

// All class can share the same creator as a function template
template <typename BaseClassName, typename SubClassName>
BaseClassName* ClassRegistry_NewObject() {
  return new SubClassName();
}

template <typename BaseClassName>
class HAMI_EXPORT ClassRegister{public : ClassRegister(
    typename ClassRegistryBase<BaseClassName>::ObjectGetter getterraw, std::string class_name,
    std::initializer_list<std::string> names){hami_load();
if (names.size() >= 2) {
  ClassRegistryInstance<BaseClassName>().DoAddClass(*(names.begin() + 1), getterraw);
} else if (names.size() == 1) {
  ClassRegistryInstance<BaseClassName>().DoAddClass(class_name, getterraw);
}
if (names.size() >= 3) {
  printlog_and_throw("ClassRegister: too many parameters.");
  // ClassRegistryInstance<BaseClassName>().DoAddClass(names[2], getterraw);
}
}  // namespace hami

~ClassRegister() {}
}
;  // namespace hami

}  // namespace hami

// class_name should not contain '>' '<' and ':'.
#define HAMI_REGISTER(base_class_type, class_name, ...)                        \
  static hami::ClassRegister<base_class_type> class_name##RegistryTag(         \
      hami::ClassRegistry_NewObject<base_class_type, class_name>, #class_name, \
      {#class_name, ##__VA_ARGS__});

#define HAMI_INSTANCE_REGISTER(base_class_type, register_name, backend_instance) \
  hami::ClassRegistryInstance<base_class_type>().DoRegisterObject(register_name, backend_instance)

#define HAMI_INSTANCE_UNREGISTER(base_class_type, register_name) \
  hami::ClassRegistryInstance<base_class_type>().DoUnRegisterObject(register_name)

#define HAMI_INSTANCE_CLEANUP(base_class_type) \
  hami::ClassRegistryInstance<base_class_type>().DoCleanUp()

// #define HAMI_CREATE(base_class_type, register_name)
//   hami::ClassRegistryInstance<base_class_type>().DoGetObject(register_name)

#define HAMI_CREATE(base_class_type, register_name, ...) \
  hami::ClassRegistryInstance<base_class_type>().DoCreateObject(register_name, ##__VA_ARGS__)

#define HAMI_INSTANCE_GET(base_class_type, aspect_name) \
  hami::ClassRegistryInstance<base_class_type>().DoGetObject(aspect_name)

// #define HAMI_GET_REGISTER_NAME(base_class_type, obj_ptr)
//   hami::ClassRegistryInstance<base_class_type>().GetObjectName(obj_ptr)

#define HAMI_OBJECT_NAME(base_class_type, obj_ptr) \
  hami::ClassRegistryInstance<base_class_type>().GetObjectName(obj_ptr)

#define HAMI_ALL_NAMES(base_class_type) hami::ClassRegistryInstance<base_class_type>().HamiGetAll()

/// ************************************************************************************************

#define GENERATE_BACKEND(aspect_class_type, derived_aspect_cls, config_setting) \
  class derived_aspect_cls : public aspect_class_type {                         \
   public:                                                                      \
    void init(const std::unordered_map<string, string>& config,                 \
              const dict& dict_config) override {                               \
      auto new_config = config;                                                 \
      auto config_addin = multi_str_split(config_setting, '=', '/');            \
      for (const auto& item : config_addin) {                                   \
        new_config.emplace(item[0], item[1]);                                   \
      }                                                                         \
      this->aspect_class_type::init(new_config, dict_config);                   \
    }                                                                           \
  };                                                                            \
  HAMI_REGISTER(Backend, derived_aspect_cls);

#endif
