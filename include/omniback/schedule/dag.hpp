#pragma once

#include <set>

#include "omniback/core/backend.hpp"
#include "omniback/core/event.hpp"
#include "omniback/core/parser.hpp"
#include "omniback/helper/threadsafe_queue.hpp"
namespace omniback {
class DagDispatcher : public HasEventForwardGuard {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict& kwargs) override final;

 public:
  void evented_forward(const std::vector<dict>& input_output) override final;

 public:
  ~DagDispatcher() {
    bInited_.store(false);
    for (auto& item : task_queues_) {
      item->notify_all();
    }
    for (auto& one_thread : threads_)
      if (one_thread.joinable()) {
        one_thread.join();
      }
  }

 protected:
  std::unordered_map<std::string, Backend*> base_dependencies_;
  // str::mapmap base_config_;

 protected:
  struct Stack {
    dict input_data;
    std::shared_ptr<Event> input_event;
    std::size_t task_queue_index = 0;
    std::exception_ptr exception;
    // std::string curr_node_name;

    struct Dag {
      std::unordered_map<std::string, dict> processed;
      std::unordered_set<std::string> waiting_nodes;
      size_t total{0};
      // std::unordered_set<std::string> processed_or_processing_nodes;
    };
    Dag dag;
    // dag specific
  };
  void clear(Stack* pstack) {
    pstack->dag.processed.clear();
    // pstack->dag.processed_or_processing_nodes.clear();
    pstack->input_event = nullptr;
    pstack->input_data = nullptr;
  }

 private:
  std::vector<std::unique_ptr<Backend>> owned_backends_;
  std::unique_ptr<parser::DagParser> dag_parser_;

 private:
  std::vector<std::unique_ptr<ThreadSafeQueue<dict>>> task_queues_;
  std::vector<std::thread> threads_;
  std::atomic_bool bInited_{true};
  void task_loop(std::size_t thread_index, ThreadSafeQueue<dict>* pqueue);

  std::string on_start_node(const dict& tmp_data, std::size_t task_queue_index);
  void on_start_nodes(
      const std::vector<dict>& tmp_data,
      std::size_t task_queue_index);
  void on_finish_node(dict tmp_data, std::shared_ptr<Stack> pstack);

  void map_or_filter_data(std::string node_name, std::shared_ptr<Stack> pstack);
  void execute(
      std::string node_name,
      std::shared_ptr<Stack> pstack,
      dict curr_data);
};
} // namespace omniback