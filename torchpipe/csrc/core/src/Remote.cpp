// Copyright 2021-2024 NetEase.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Remote.hpp"

#include "subprocess/subprocess.h"

#include <memory>
#include <vector>
#include "base_logging.hpp"
#include "params.hpp"
#include <fstream>
#include "reflect.h"
#include "Serialize.hpp"
namespace ipipe {

bool Remote::init(const std::unordered_map<std::string, std::string>& config_param,
                  dict dict_config) {
  auto data = serialize(config_param);
  const char* command_line[] = {"echo", "\"${PATH}\"", NULL};
  // const char* command_line[] = {"pwd", NULL};
  struct subprocess_s subprocess;
  auto opt = subprocess_option_no_window | subprocess_option_enable_async |
             subprocess_option_search_user_path;

  int result = subprocess_create(command_line, opt, &subprocess);
  IPIPE_ASSERT(0 == result);
  if (0 == result) {
    std::string buffer(1024, '\0');
    std::string data;
    unsigned bytesRead = 0;
    int ret = EXIT_SUCCESS;
    do {
      bytesRead = subprocess_read_stdout(&subprocess, &buffer[0], buffer.size());
      data += buffer.substr(0, bytesRead);
    } while (bytesRead != 0);

    subprocess_join(&subprocess, &ret);
    subprocess_destroy(&subprocess);

    printf("%s", data.c_str());
  }

  return true;

  // params_ = std::unique_ptr<Params>(new Params({}, {"resize_h", "resize_w"}, {}, {}));
  // if (!params_->init(config_param)) return false;

  // LOG_EXCEPTION(resize_h_ = std::stoi(params_->operator[]("resize_h")));
  // LOG_EXCEPTION(resize_w_ = std::stoi(params_->operator[]("resize_w")));
  // if (resize_h_ > 1024 * 1024 || resize_w_ > 1024 * 1024 || resize_h_ < 1 || resize_w_ < 1) {
  //   SPDLOG_ERROR("Remote: illigle h or w: h=" + std::to_string(resize_h_) +
  //                "w=" + std::to_string(resize_w_));
  //   return false;

  return true;
}

void Remote::forward(dict input_dict) {
  auto& input = *input_dict;

  return;
}
IPIPE_REGISTER(Backend, Remote, "Remote");
}  // namespace ipipe