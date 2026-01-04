
#pragma once
#include <omniback/extension.hpp>
#include <opencv2/core/types.hpp>
namespace torchpipe {

class ResizeMat : public omniback::BackendOne {
 public:
  virtual void impl_init(
      const std::unordered_map<std::string, std::string>&,
      const omniback::dict&) override;

  virtual void forward(const omniback::dict&) override;

 private:
  size_t resize_h_;
  size_t resize_w_;
};

/**
 * Resizes an image while maintaining aspect ratio (letterbox style)
 * - Centers the resized image in the target canvas
 * - Fills empty areas with specified padding color
 * - Outputs scaling factor and offset values for coordinate mapping
 */
class LetterBoxMat : public omniback::BackendOne {
 public:
  virtual void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const omniback::dict& kwargs) override;

  virtual void forward(const omniback::dict& input_dict) override;

 private:
  size_t target_h_; // Target height
  size_t target_w_; // Target width
  cv::Scalar pad_val_; // Padding color (BGR)
};

/**
 * Resizes an image while maintaining aspect ratio (top-left aligned)
 * - Places resized image at top-left corner
 * - Fills remaining areas with padding color
 * - Outputs scaling factor (offset always 0,0)
 */
class TopLeftResizeMat : public omniback::BackendOne {
 public:
  virtual void impl_init(
      const std::unordered_map<std::string, std::string>& config,
      const omniback::dict& kwargs) override;

  virtual void forward(const omniback::dict& input_dict) override;

 private:
  size_t target_h_;
  size_t target_w_;
  cv::Scalar pad_val_;
};

} // namespace torchpipe