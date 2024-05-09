#include "Interpreter.hpp"
#include <fstream>
#include "opencv2/core.hpp"

void main() {
  auto input = std::make_shared<std::unordered_map<std::string, ipipe::any>>();
  std::ifstream jpg("image.jpg", std::ios::binary);
  if (jpg.is_open()) {
    std::string data((std::istreambuf_iterator<char>(jpg)), std::istreambuf_iterator<char>());

    (*input)[ipipe::TASK_DATA_KEY] = data;
  }

  ipipe::Interpreter interpreter;
  std::unordered_map<std::string, std::string> config;
  config["backend"] = "DecodeMat";
  interpreter.init(config);
  interpreter.forward({input});

  auto iter = input->find(ipipe::TASK_RESULT_KEY);
  if (iter == input->end()) {
    std::cout << "[error] no result,   " << std::endl;
  } else {
  }

  cv::Mat img = ipipe::any_cast<cv::Mat>(iter->second);
}
