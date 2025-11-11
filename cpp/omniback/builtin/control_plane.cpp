
#include "omniback/builtin/control_plane.hpp"
#include <string>
#include <vector>
#include "omniback/core/parser.hpp"

namespace omniback {
void ControlPlane::impl_init(
    const std::unordered_map<std::string, std::string>& params,
    const dict& options) {
  //   std::string cls_name = default_cls_name();
  auto cls_name = parser_v2::get_backend_name(this);

  parser_v2::Parser parser;

  auto iter = params.find(cls_name + "::dependency");
  OMNI_ASSERT(
      iter != params.end(),
      "Dependency configuration " + cls_name +
          "::dependency not found. "
          "This control backend do not allow runtime dynamic modification of "
          "dependencies, "
          "please specify dependencies in the initializtion phase");

  std::pair<std::vector<char>, std::vector<std::string>> sub_config =
      parser.split_by_delimiters(iter->second, ',', ';');
  delimiters_ = sub_config.first;
  OMNI_ASSERT(
      sub_config.second.size() >= 1, "backend_names.size() should >= 1");

  for (auto sub_iter = sub_config.second.begin();
       sub_iter != sub_config.second.end();
       ++sub_iter) {
    std::pair<std::string, std::string> sub_cfg =
        parser.prifix_split(*sub_iter, '(', ')');

    auto [args, str_kwargs] = parser.parse_args_kwargs(sub_cfg.first);
    std::unordered_map<std::string, std::string> update_params;
    parser::update(params, str_kwargs);
    backend_cfgs_.emplace_back(sub_cfg.second);
    prefix_args_kwargs_.push_back({args, str_kwargs});
    main_backends_.emplace_back(sub_cfg.second.substr(
        0, std::min(sub_cfg.second.find('('), sub_cfg.second.find('['))));
  }
  OMNI_ASSERT(!backend_cfgs_.empty());

  impl_custom_init(params, options);
  update_min_max();
}
} // namespace omniback