#include <string>
#include <unordered_map>
#include <memory>

#include <torch/torch.h>

#include "Streaming.hpp"
#include "Backend.hpp"
#include "params.hpp"
#include "exception.hpp"
#include "threadsafe_queue.hpp"
#include "threadsafe_kv_storage.hpp"
namespace ipipe {
/**
 * @brief Streaming class that extends SingleBackend.
 *
 * This class represents a streaming backend for the TorchPipe library.
 * It implements the init() and forward() methods required by the SingleBackend interface.
 */
class Streaming : public SingleBackend {
 public:
  /**
   * @brief Initializes the streaming backend with the given configuration parameters.
   *
   * @param config_param The configuration parameters for the backend.
   * @param dict The dictionary containing additional parameters.
   * @return True if initialization is successful, false otherwise.
   */
  virtual bool init(const std::unordered_map<std::string, std::string> &config_param,
                    dict) override {
    params_ = std::unique_ptr<Params>(new Params({{"top_k", "50"}, {"top_p", "0.2"}}, {}, {}, {}));
    if (!params_->init(config_param)) return false;

    auto topp_ = std::stof(params_->at("top_p"));
    return true;
  }

  /**
   * @brief Performs the forward pass of the streaming backend.
   *
   * @param input_dict The input dictionary containing the input data.
   */
  virtual void forward(dict input_dict) override {
    auto &input = *input_dict;
    auto iter = input.find("request_id");
    IPIPE_ASSERT(iter != input.end(), "request_id not found in input_dict");
    std::string request_id = any_cast<std::string>(iter->second);
    long tensor_item = RETURN_EXCEPTION_TRACE(any_cast<long>(input.at("tensor_item")));

    std::shared_ptr<ThreadSafeQueue<long>> obj;
    auto &ins = ThreadSafeKVStorage::getInstance();
    auto que = ins.get(request_id, "queue");
    if (que) {
      obj = any_cast<std::shared_ptr<ThreadSafeQueue<long>>>(*que);
      obj->Push(tensor_item);
    } else {
      obj = std::make_shared<ThreadSafeQueue<long>>();
      obj->Push(tensor_item);
      ins.set(request_id, "queue", obj);
    }

    if (input.find("is_eos") != input.end()) {
      ins.set(request_id, "is_eos", input["is_eos"]);
    }

    input[TASK_RESULT_KEY] = input[TASK_DATA_KEY];
  }

 private:
  std::unique_ptr<Params> params_;
  std::mutex mutex_;
};

IPIPE_REGISTER(Backend, Streaming, "Streaming")
};  // namespace ipipe

// #ifdef PYBIND
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>

// namespace py = pybind11;

// template<typename T>
// void bind_ThreadSafeQueue(py::module &m, const std::string &typestr) {
//     using QueueType = ThreadSafeQueue<T>;
//     using QueueTypePtr = std::shared_ptr<QueueType>;

//     py::class_<QueueType, QueueTypePtr>(m, ("ThreadSafeQueue" + typestr).c_str())
//         .def(py::init<>())
//         .def("push", &QueueType::push)
//         .def("pop", &QueueType::pop);
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     bind_ThreadSafeQueue<long>(m, "Long");
//     // Add more types if needed
// }
// #endif