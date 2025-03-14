#include "hami/builtin/condition.hpp"
#include "hami/helper/string.hpp"
#include "hami/helper/base_logging.hpp"
namespace hami {

void Condition::pre_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& kwargs) {
    condition_ = condition_impl();
    init_dep_impl(config, kwargs);
}

class HasKey : public Condition {
   public:
    void init_dep_impl(
        const std::unordered_map<std::string, std::string>& config,
        const dict& kwargs) override {
        if (config.find("key") != config.end()) {
            key_ = config.at("key");
        }
    }

    virtual std::function<bool(const dict&)> condition_impl() const override {
        return [this](const dict& input) {
            return input->find(key_) != input->end();
        };
    }

   protected:
    std::string key_{"key"};
};

class NotHasKey : public HasKey {
    virtual std::function<bool(const dict&)> condition_impl() const override {
        return [this](const dict& input) {
            return input->find(key_) == input->end();
        };
    }
};

HAMI_REGISTER(Backend, HasKey);
HAMI_REGISTER(Backend, NotHasKey);

}  // namespace hami