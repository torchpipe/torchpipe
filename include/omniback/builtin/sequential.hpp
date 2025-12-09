#pragma once

#include <string>
#include <vector>

#include "omniback/builtin/basic_backends.hpp"
#include "omniback/builtin/control_plane.hpp"
#include "omniback/core/backend.hpp"
namespace omniback {

class SequentialV0 final : public Container {
 public:
  /**
   * @param SequentialV0::dependency The name of the sub-backend, multiple
   * backends separated by commas.
   * @remark
   * 1. The initialization of sub-backends will be executed in **reverse
   * order**;
   * 2. This container supports expanding bracket compound syntax. It use the
   * @ref str::brackets_split function to expand it as follows:
   * - backend = B[C]        =>     {backend=B, B::dependency=C}
   * - backend = D           =>     {backend=D}
   * - backend = B[E[Z1,Z2]] =>     {backend=B, B::dependency=E[Z1,Z2]}
   */
  virtual void post_init(
      const std::unordered_map<std::string, std::string>&,
      const dict&) override {}

  /**
   * @brief SequentialV0ly calls sub-backends.
   * @note During sequential calls, a swap operation will be performed: Assign
   * TASK_RESULT_KEY to TASK_DATA_KEY.
   */
  virtual void impl_forward(const std::vector<dict>&) override;
};

/**
 * @class Sequential
 * @brief A class representing a sequential backend.
 *
 * This class sequentially calls sub-backends and performs a swap operation
 * during the calls.
 */
class Sequential final : public ControlPlane {
 private:
  /**
   * @brief Initializes the sub-backends.
   *
   * This method initializes the sub-backends in reverse order. The names of
   * the sub-backends are specified using the `SequentialV0::dependency`
   * parameter, which supports multiple backends separated by commas. It also
   * supports an or prefix filter.
   *
   * @param params A map of parameters for initialization.
   * @param options A dictionary of options.
   */
  virtual void impl_custom_init(
      const std::unordered_map<std::string, std::string>& params,
      const dict& options) override;

  /**
   * @brief SequentialV0ly calls sub-backends.
   *
   * @param io A vector of input/output dictionaries for each sub-backend
   * call.
   */
  virtual void impl_forward(const std::vector<dict>& io) override;
  void update_min_max() override;

 private:
  std::vector<bool> filter_or_;
  std::vector<std::unique_ptr<Backend>> backends_;
  // size_t min_{1};
  // size_t max_{std::numeric_limits<std::size_t>::max()};
};

} // namespace omniback