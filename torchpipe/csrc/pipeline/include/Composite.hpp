#include <memory>
#include <vector>
#include <atomic>
#include <thread>

#include "threadsafe_queue.hpp"
#include "Backend.hpp"
#include "event.hpp"
#include "params.hpp"
#pragma once
namespace ipipe {
class Composite : public Backend {
 public:
  bool init(const std::unordered_map<std::string, std::string>& config, dict) override;
  void forward(const std::vector<dict>& inputs) override;

 protected:
 private:
  // void forward(dict input, std::size_t task_queues_index, std::shared_ptr<SimpleEvents>
  // curr_event);

 private:
  std::unique_ptr<Backend> backend_;

  std::unique_ptr<Params> params_;

  std::vector<std::string> engine_names_;
};
}  // namespace ipipe
