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

#pragma once

#include <unordered_map>
#include <vector>
#include "dict.hpp"
namespace ipipe {

/**
 * @brief 所有后端的基类。
 */
class Backend {
 public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  Backend() = default;
  Backend(const Backend&) = delete;
  Backend(const Backend&&) = delete;
  Backend& operator=(const Backend&) = delete;
  Backend& operator=(const Backend&&) = delete;
  virtual ~Backend() = default;
#endif

  /**
   * @if chinese
   * @brief 初始化函数；不可重复初始化。
   * @param config 输入的参数。值类型通常为str或者int，float
   * @param dict_config 输入的任意类型的参数
   * @else
   * @brief initialisation
   * @note  sub-class method is allowed to throw any exception.
   * @param config input paramaters.（key: str, value: str int, float)
   * @param dict_config input paramaters.（key: str, value: any)
   * @endif
   */
  virtual bool init(const std::unordered_map<std::string, std::string>& config, dict dict_config) {
    return true;
  };

  /**
   * @if chinese
   * @brief 前向接口.
   * @param  input_dicts 输入输出数据。输入键值必定包含 TASK_DATA_KEY ,
   * 输出如果不含 TASK_RESULT_KEY，调用方可认为执行出现异常。
   * @note 以下由调用方保证：输入数据包含TASK_DATA_KEY，且调用时
   * 初始化已经完成；
   * @else
   * not finished
   * @endif
   */
  virtual void forward(const std::vector<dict>& input_dicts) = 0;

  /**
   * @if chinese
   * @return 1. 前向接口可接受的最大数据量。 范围是[1, UINT32_MAX],
   * 调度后端和一些功能性后端等特殊情况可以设置为UINT32_MAX。其他情况建议按照实际运算并行方式去设置。比如，一般cpu后端，建议设置为1，也就是继承
   * @ref SingleBackend 即可。
   * @remark 类的实例初始化后有效，不同实例可以返回不同值。
   * @else
   * @endif
   */
  virtual uint32_t max() const { return 1; };
  /**
   * @if chinese
   * @return 1. 前向接口可接受的最小数据量。
   * @else
   * @endif
   */
  virtual uint32_t min() const { return 1; };

  // /**
  //  * @if chinese
  //  * @return 是否分裂
  //  * @else
  //  * @endif
  //  */
  // virtual bool split(dict data, std::vector<dict>& out) const { return false; };
};

/**
 * @brief 输入范围为[1, 1], 继承此基类的后端只需要实现单个dict的前向处理。
 *
 */
class SingleBackend : public Backend {
 public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  virtual void forward(const std::vector<dict>& input_dicts) override final {
    if (input_dicts.size() != 1)
      throw std::invalid_argument("SingleBackend: error input_dicts.size()=" +
                                  std::to_string(input_dicts.size()));
    if (input_dicts[0]->find(TASK_DATA_KEY) == input_dicts[0]->end()) {
      throw std::runtime_error("the input data should contain TASK_DATA_KEY");
    }
    forward(input_dicts[0]);
  };
#endif

  /**
   * 子类重新实现了此函数。
   */
  virtual void forward(dict input_dict) = 0;

  /// @return 1
  virtual uint32_t max() const override final { return 1; };
};

class EmptyForwardSingleBackend : public SingleBackend {
  virtual void forward(dict input_dict) override final{};
};

}  // namespace ipipe
