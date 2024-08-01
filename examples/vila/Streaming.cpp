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
   * @brief Performs the forward pass of the streaming backend.
   *
   * @param input_dict The input dictionary containing the input data.
   */
  virtual void forward(dict input_dict) override {
    auto &input = *input_dict;
    auto iter = input.find("request_id");
    IPIPE_ASSERT(iter != input.end(), "request_id not found in input_dict");
    std::string request_id = any_cast<std::string>(iter->second);
    long tensor_item = RETURN_EXCEPTION_TRACE(any_cast<long>(input.at(TASK_CPU_RESULT_KEY)));

    std::shared_ptr<ThreadSafeQueue<long>> obj;
    static auto &ins = ThreadSafeKVStorage::getInstance();
    auto &storage_kv = ins.get(request_id);
    auto que = storage_kv.get("queue");
    if (que) {
      obj = any_cast<std::shared_ptr<ThreadSafeQueue<long>>>(*que);
      obj->Push(tensor_item);
    } else {
      obj = std::make_shared<ThreadSafeQueue<long>>();
      obj->Push(tensor_item);
      storage_kv.set("queue", obj);
    }

    input[TASK_RESULT_KEY] = input[TASK_DATA_KEY];
  }

 private:
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