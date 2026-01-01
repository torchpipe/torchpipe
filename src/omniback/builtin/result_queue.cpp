#include "omniback/builtin/result_queue.hpp"

#include <thread>

#include "omniback/builtin/basic_backends.hpp"
#include "omniback/core/helper.hpp"
#include "omniback/core/parser.hpp"
#include "omniback/helper/base_logging.hpp"
#include "omniback/helper/string.hpp"
#include "omniback/helper/timer.hpp"

namespace omniback {

// init = List[QueueBackend[register_name, optional[target_name]]]
void QueueBackend::pre_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict&) {
  auto dep = get_dependency_name_force(this, config);
  auto iter = dep.find(',');

  if (iter == std::string::npos) {
    register_name_ = dep;
  } else {
    register_name_ = dep.substr(0, iter);
    target_name_ = dep.substr(iter + 1);
  }
  OMNI_ASSERT(
      !register_name_.empty(),
      "QueueBackend should have register name: "
      "Queue[register_name, optional[target_name]");
  OMNI_INSTANCE_REGISTER(Backend, register_name_, this);
  OMNI_ASSERT(owned_queue_);
  queue_ = owned_queue_.get();
  OMNI_INSTANCE_REGISTER(Queue, register_name_, queue_);

  if (!target_name_.empty()) {
    target_backend_ = OMNI_INSTANCE_GET(Backend, target_name_);
    OMNI_ASSERT(target_backend_);
  }
}

void QueueBackend::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict& kwargs) {
  pre_init(config, kwargs);
  if (target_backend_) {
    bInited_ = true;

    thread_ = std::thread(&QueueBackend::run, this);
  }
}

void QueueBackend::impl_inject_dependency(Backend* dep) {
  if (target_backend_) {
    target_backend_->inject_dependency(dep);
  } else {
    target_backend_ = dep;
  }
}

void QueueBackend::run() {
  while (bInited_.load()) {
    auto data = queue_->try_get<omniback::dict>(SHUTDOWN_TIMEOUT);
    if (!data.has_value())
      continue;
    auto io_data = (data.value());
    assert(io_data);
    target_backend_->forward({io_data});
  }
}

QueueBackend::~QueueBackend() {
  bInited_ = false;
  if (thread_.joinable()) {
    thread_.join();
  }
}

OMNI_REGISTER_BACKEND(QueueBackend, "QueueBackend,AsyncQueue, Queue");

// SrcFromQueue
// init = Register[QueueBackend[x]]
// QueueBackend[register_name, target_name]

// init = List[Send[target_name]]
void Send::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict&) {
  auto target_name = get_dependency_name(this, config);
  OMNI_ASSERT(target_name);

  auto [args, str_kwargs] = parser::parse_args_kwargs(*target_name);
  parser::update(config, str_kwargs);
  str::try_update(str_kwargs, "max", queue_max_);

  OMNI_ASSERT(
      args.size() <= 1, "Send must have at most one queue instance name");

  if (!args.empty())
    queue_ = OMNI_INSTANCE_GET(Queue, args[0]);
  else {
    queue_ = &(default_queue());
  }
  OMNI_ASSERT(queue_);
}

void Send::impl_forward(const std::vector<dict>& input) {
  while (!queue_->try_pushes(input, queue_max_, SHUTDOWN_TIMEOUT)) {
  };
  // for (auto& item : input) {
  //     queue_->put(item);
  // }
}

class Send2Queue : public BackendOne {
 public:
  ~Send2Queue() {
    // queue_->cancel();
  }

 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict&) override final {
    auto [args, kwargs] =
        parser_v2::get_args_kwargs(this, "Send2Queue", config);

    str::try_update(kwargs, "queue_max", queue_max_);
    str::try_update(kwargs, "keep_result", keep_result_);
    if (args.size() == 1) {
      queue_ = &(default_queue(args[0]));
      SPDLOG_INFO("Using default queue: {}", args[0]);
    } else if (args.empty()) {
      queue_ = &(default_queue());
    } else {
      OMNI_ASSERT(
          false, "Send2Queue must have at most one queue instance name");
    }

    OMNI_ASSERT(queue_);
  }

  void forward(const dict& input) override {
    auto data = copy_dict(input);
    while (!queue_->try_push(data, queue_max_, SHUTDOWN_TIMEOUT)) {
    };
    // SPDLOG_INFO("Send2Queue {} ", queue_->size());
    if (keep_result_) {
      (*input)[TASK_RESULT_KEY] = input->at(TASK_DATA_KEY);
    }
  }

 protected:
  Queue* queue_{nullptr};
  size_t queue_max_{std::numeric_limits<uint32_t>::max()};
  int keep_result_{1};
};

class SrcQueue : public BackendOne {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict&) override final {
    auto [args, kwargs] = parser_v2::get_args_kwargs(this, "SrcQueue", config);

    str::try_update(kwargs, "max", queue_max_);
    if (args.size() == 1) {
      queue_ = OMNI_INSTANCE_GET(Queue, args[0]);
    } else if (args.empty()) {
      queue_ = &(default_queue());
    } else {
      OMNI_ASSERT(false, "SrcQueue must have at most one queue instance name");
    }

    OMNI_ASSERT(queue_);
  }

  void forward(const dict& input) override {
    (*input)[TASK_RESULT_KEY] = queue_;
  }

 protected:
  Queue* queue_{nullptr};
  size_t queue_max_{std::numeric_limits<uint32_t>::max()};
};

OMNI_REGISTER_BACKEND(SrcQueue);

// class Profile : public Dependency {
//     void impl_forward_with_dep(const std::vector<dict> &input_output,
//                                Backend *dep) override {}
// };

class CreateQueue : public Backend {
 private:
  void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const dict&) override final {
    auto [args, kwargs] =
        parser_v2::get_args_kwargs(this, "CreateQueue", config);

    OMNI_ASSERT(args.size() == 1, "Usage: CreateQueue(register_name)");
    // OMNI_INSTANCE_REGISTER(Queue, args[0], owned_queue_.get());
    queue_ = &default_queue(args[0]);

    str::try_update(kwargs, "max", queue_max_);
  }

  void impl_forward(const std::vector<dict>& input) override {
    while (!queue_->try_pushes(input, queue_max_, SHUTDOWN_TIMEOUT)) {
    };
  }

 protected:
  Queue* queue_{nullptr};
  std::unique_ptr<Queue> owned_queue_{std::make_unique<Queue>()};
  size_t queue_max_{std::numeric_limits<uint32_t>::max()};
};

OMNI_REGISTER_BACKEND(Send2Queue, "Send2Queue");
OMNI_REGISTER_BACKEND(CreateQueue, "CreateQueue");

void Observer::impl_forward(const std::vector<dict>& input) {
  for (auto& item : input) {
    (*item)[TASK_RESULT_KEY] = item->at(TASK_DATA_KEY);
    auto new_dict = copy_dict(item);
    queue_->push(new_dict);
  }
}

OMNI_REGISTER_BACKEND(Send, "Send");
OMNI_REGISTER_BACKEND(Observer);

// init = List[Recv[register_name_for_src_que, target_backend_name]]
void Recv::pre_init(
    const std::unordered_map<std::string, std::string>& config,
    const dict&) {
  auto dep = get_dependency_name_force(this, config);
  auto iter = dep.find(',');

  OMNI_ASSERT(
      iter != std::string::npos,
      "Usage: Recv[src_queue_name, target_backend_name]");

  auto register_name = dep.substr(0, iter);
  auto target_name = dep.substr(iter + 1);

  OMNI_ASSERT(
      !register_name.empty() && !target_name.empty(),
      "Recv must have register name and target name");
  queue_ = OMNI_INSTANCE_GET(Queue, register_name);

  target_backend_ = OMNI_INSTANCE_GET(Backend, target_name);
  OMNI_ASSERT(target_backend_);
}

} // namespace omniback