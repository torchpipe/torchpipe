
#include <algorithm>
#include <cmath>

#include "hami/builtin/basic_backends.hpp"
#include "hami/core/reflect.h"
#include "hami/helper/macro.h"
#include "hami/helper/base_logging.hpp"
#include "hami/core/task_keys.hpp"
#include "hami/core/helper.hpp"
#include "hami/helper/string.hpp"

namespace hami {

class Add : public SingleBackend {
   public:
    void impl_init(const std::unordered_map<std::string, std::string>& config,
                   const dict& dict_config) override {
        std::string dep = get_dependency_name_force(this, config);
        keys_ = str::map_split(dep, ':', ',');
        try_replace_inner_key(keys_);
        has_result_ = keys_.find(TASK_RESULT_KEY) != keys_.end();
    }
    void forward(const dict& input_dict) override {
        for (const auto& item : keys_) {
            (*input_dict)[item.first] = item.second;
        }
        if (!has_result_)
            (*input_dict)[TASK_RESULT_KEY] = input_dict->at(TASK_DATA_KEY);
    }

   private:
    std::unordered_map<std::string, std::string> keys_;
    bool has_result_{false};
};
HAMI_REGISTER(Backend, Add);

class AddInt : public SingleBackend {
   public:
    void impl_init(const std::unordered_map<std::string, std::string>& config,
                   const dict& dict_config) override {
        std::string dep = get_dependency_name_force(this, config);
        auto keys = str::map_split(dep, ':', ',');
        try_replace_inner_key(keys);
        has_result_ = keys.find(TASK_RESULT_KEY) != keys.end();
        for (const auto& item : keys) {
            keys_[item.first] = std::stoi(item.second);
        }
    }
    void forward(const dict& input_dict) override {
        for (const auto& item : keys_) {
            (*input_dict)[item.first] = item.second;
        }
        if (!has_result_)
            (*input_dict)[TASK_RESULT_KEY] = input_dict->at(TASK_DATA_KEY);
    }

   private:
    std::unordered_map<std::string, int> keys_;
    bool has_result_{false};
};
HAMI_REGISTER(Backend, AddInt);

class Remove : public SingleBackend {
   public:
    void impl_init(const std::unordered_map<std::string, std::string>& config,
                   const dict& dict_config) override {
        std::string dep = get_dependency_name_force(this, config);
        auto keys = str::str_split(dep, ',');
        for (auto key : keys_) {
            try_replace_inner_key(key);
            keys_.insert(key);
        }

        has_result_ = keys_.find(TASK_RESULT_KEY) != keys_.end();
    }
    void forward(const dict& input_dict) {
        for (const auto& item : keys_) {
            input_dict->erase(item);
        }
        if (!has_result_)
            (*input_dict)[TASK_RESULT_KEY] = input_dict->at(TASK_DATA_KEY);
    }

   private:
    std::unordered_set<std::string> keys_;
    bool has_result_{false};
};
HAMI_REGISTER(Backend, Remove);

}  // namespace hami