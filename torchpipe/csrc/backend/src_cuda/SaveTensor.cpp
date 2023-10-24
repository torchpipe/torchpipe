// Copyright 2021-2023 NetEase.
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

#include <torch/serialize.h>
#include <ATen/ATen.h>
#include <fstream>
#include "base_logging.hpp"
#include "reflect.h"
#include "dict.hpp"
#include "params.hpp"
#include "file_utils.hpp"

namespace {
inline const std::string thread_id_string() {
  std::stringstream ss;
  ss << std::this_thread::get_id();
  return ss.str();
}
};  // namespace
namespace ipipe {

/**
 * @brief 用于保存tensor到磁盘(.pt文件)，可在python中使用 ``torch.load()`` 加载.
 * 如果想保存为图片，可连续使用
 * @ref Tensor2Mat, @ref SaveMat.
 * @warning 会严重影响性能，只能用做调试；
 */
class SaveTensor : public SingleBackend {
 public:
  /**
   * @param save_dir 文件保存目录，需要提前创建；
   * @param node_name
   * 可选参数，节点名称，用作文件名的一部分。此部分会由调度后端自动填充,
   * 不可人为填入。
   *
   */
  virtual bool init(const std::unordered_map<std::string, std::string>& config_param,
                    dict) override {
    params_ = std::unique_ptr<Params>(new Params({{"node_name", ""}}, {"save_dir"}, {}, {}));
    if (!params_->init(config_param)) return false;

    save_dir_ = params_->at("save_dir");
    std::ifstream file(save_dir_.c_str());

    if (save_dir_.empty() && !file.good()) {
      SPDLOG_ERROR("SaveTensor: dir " + save_dir_ + " not exists.");
      return false;
    }
    if (!os_path_exists(save_dir_)) {
      SPDLOG_ERROR("SaveTensor: dir " + save_dir_ + " not exists. Please try `mkdir {}.`",
                   save_dir_);
      return false;
    }
    return true;
  }

  /**
   * @brief 命名由线程id，thread_local 的 index 决定，故进程内维一
   * @param TASK_RESULT_KEY input[TASK_RESULT_KEY] = input[TASK_DATA_KEY]
   */
  virtual void forward(dict input_dict) override {
    auto& input = *input_dict;
    if (input[TASK_DATA_KEY].type() != typeid(at::Tensor)) {
      SPDLOG_ERROR("SaveTensor: at::Tensor needed; error input type: " +
                   std::string(input[TASK_DATA_KEY].type().name()));
      throw std::runtime_error("SaveTensor: at::Tensor needed; error input type: " +
                               std::string(input[TASK_DATA_KEY].type().name()));
    }
    auto input_tensor = any_cast<at::Tensor>(input[TASK_DATA_KEY]);

    const std::string& save_dir = params_->at("save_dir");

    thread_local int index_ = 0;

    thread_local const auto base_save_name =
        save_dir + "/" + params_->get("node_name", "") + "_" + thread_id_string() + "_";

    auto save_name = base_save_name + std::to_string(index_) + ".pt";
    // imwrite(save_name, input_tensor);
    std::vector<char> data_for_save = torch::pickle_save(input_tensor);
    // save_name
    std::ofstream fout(save_name, std::ios::out | std::ios::binary);
    fout.write(data_for_save.data(), data_for_save.size());
    fout.close();
    index_++;
    SPDLOG_WARN("image dumped for debug: " + save_name +
                " . Note that dumping affect the performance.");

    input[TASK_RESULT_KEY] = input[TASK_DATA_KEY];
  }

 private:
  std::unique_ptr<Params> params_;
  std::string save_dir_;
};

IPIPE_REGISTER(Backend, SaveTensor, "SaveTensor");

}  // namespace ipipe