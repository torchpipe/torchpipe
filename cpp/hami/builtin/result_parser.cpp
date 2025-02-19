#include "hami/helper/string.hpp"
#include "hami/helper/base_logging.hpp"
#include "hami/builtin/result_parser.hpp"
#include "hami/core/task_keys.hpp"
#include "hami/core/reflect.h"

namespace hami {

void ResultParser::pre_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& dict_config) {
    parser_ = parser_impl();
    init_impl(config, dict_config);
}

class ThrowIfNoResult : public ResultParser {
   public:
    virtual std::function<void(const dict& data)> parser_impl() const override {
        return [](const dict& input) {
            if (input->find(TASK_RESULT_KEY) == input->end()) {
                throw std::runtime_error("ThrowIfNoResult: No result found");
            }
        };
    }
};

HAMI_REGISTER(Backend, ThrowIfNoResult);

}  // namespace hami