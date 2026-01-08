
#pragma once
#include <omniback/extension.hpp>
namespace torchpipe {

class Mat2Tensor : public om::BackendOne {
 public:
  virtual void impl_init(
      const std::unordered_map<std::string, std::string>&,
      const om::dict&) override;

  virtual void forward(const om::dict&) override;

 private:
  std::string device_{"cuda"};
};

class Tensor2Mat : public om::BackendOne {
 public:
  //   virtual void impl_init(
  //       const std::unordered_map<std::string, std::string>&,
  //       const om::dict&) override;

  virtual void forward(const om::dict&) override;

 private:
};

// cv::Mat ImageData2Mat(const ImageData& img) ;

// convert::ImageData
//     cvMatToImageData(const cv::Mat& mat);
} // namespace torchpipe