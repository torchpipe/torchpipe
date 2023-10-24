/*
 * Copyright 2021 Zuru Tech HK Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * istributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifdef WITH_OPENCV
#include <cstdint>
#include <utility>

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include <cmath>

#include "PillowResize.hpp"

double PillowResize::BoxFilter::filter(double x) const {
  const double half_pixel = 0.5;
  if (x > -half_pixel && x <= half_pixel) {
    return 1.0;
  }
  return 0.0;
}

double PillowResize::BilinearFilter::filter(double x) const {
  if (x < 0.0) {
    x = -x;
  }
  if (x < 1.0) {
    return 1.0 - x;
  }
  return 0.0;
}

double PillowResize::HammingFilter::filter(double x) const {
  if (x < 0.0) {
    x = -x;
  }
  if (x == 0.0) {
    return 1.0;
  }
  if (x >= 1.0) {
    return 0.0;
  }
  x = x * M_PI;
  return sin(x) / x * (0.54F + 0.46F * cos(x));  // NOLINT
}

double PillowResize::BicubicFilter::filter(double x) const {
  // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
  const double a = -0.5;
  if (x < 0.0) {
    x = -x;
  }
  if (x < 1.0) {                                     // NOLINT
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;  // NOLINT
  }
  if (x < 2.0) {                             // NOLINT
    return (((x - 5) * x + 8) * x - 4) * a;  // NOLINT
  }
  return 0.0;
}

double PillowResize::LanczosFilter::_sincFilter(double x) {
  if (x == 0.0) {
    return 1.0;
  }
  x = x * M_PI;
  return sin(x) / x;
}

double PillowResize::LanczosFilter::filter(double x) const {
  // Truncated sinc.
  // According to Jim Blinn, the Lanczos kernel (with a = 3)
  // "keeps low frequencies and rejects high frequencies better
  // than any (achievable) filter we've seen so far."[3]
  // (https://en.wikipedia.org/wiki/Lanczos_resampling#Advantages)
  const double lanczos_a_param = 3.0;
  if (-lanczos_a_param <= x && x < lanczos_a_param) {
    return _sincFilter(x) * _sincFilter(x / lanczos_a_param);
  }
  return 0.0;
}

int PillowResize::_precomputeCoeffs(int in_size, double in0, double in1, int out_size,
                                    const std::shared_ptr<Filter>& filterp,
                                    std::vector<int>& bounds, std::vector<double>& kk) {
  // Prepare for horizontal stretch.
  double scale = 0;
  double filterscale = 0;
  filterscale = scale = static_cast<double>(in1 - in0) / out_size;
  if (filterscale < 1.0) {
    filterscale = 1.0;
  }

  // Determine support size (length of resampling filter).
  double support = filterp->support() * filterscale;

  // Maximum number of coeffs.
  int k_size = static_cast<int>(ceil(support)) * 2 + 1;

  // Check for overflow
  if (out_size > INT_MAX / (k_size * static_cast<int>(sizeof(double)))) {
    throw std::runtime_error("Memory error");
  }

  // Coefficient buffer.
  kk.resize(out_size * k_size);

  // Bounds vector.
  bounds.resize(out_size * 2);

  int x = 0;
  int xmin = 0;
  int xmax = 0;
  double center = 0;
  double ww = 0;
  double ss = 0;

  const double half_pixel = 0.5;
  for (int xx = 0; xx < out_size; ++xx) {
    center = in0 + (xx + half_pixel) * scale;
    ww = 0.0;
    ss = 1.0 / filterscale;
    // Round the value.
    xmin = static_cast<int>(center - support + half_pixel);
    if (xmin < 0) {
      xmin = 0;
    }
    // Round the value.
    xmax = static_cast<int>(center + support + half_pixel);
    if (xmax > in_size) {
      xmax = in_size;
    }
    xmax -= xmin;
    double* k = &kk[xx * k_size];
    for (x = 0; x < xmax; ++x) {
      double w = filterp->filter((x + xmin - center + half_pixel) * ss);
      k[x] = w;  // NOLINT
      ww += w;
    }
    for (x = 0; x < xmax; ++x) {
      if (ww != 0.0) {
        k[x] /= ww;  // NOLINT
      }
    }
    // Remaining values should stay empty if they are used despite of xmax.
    for (; x < k_size; ++x) {
      k[x] = 0;  // NOLINT
    }
    bounds[xx * 2 + 0] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return k_size;
}

std::vector<double> PillowResize::_normalizeCoeffs8bpc(const std::vector<double>& prekk) {
  std::vector<double> kk;
  kk.reserve(prekk.size());

  const double half_pixel = 0.5;
  for (const auto& k : prekk) {
    if (k < 0) {
      kk.emplace_back(static_cast<int>(-half_pixel + k * (1U << precision_bits)));
    } else {
      kk.emplace_back(static_cast<int>(half_pixel + k * (1U << precision_bits)));
    }
  }
  return kk;
}

cv::Mat PillowResize::resize(const cv::Mat& src, const cv::Size& out_size, int filter) {
  cv::Rect2f box(0.F, 0.F, static_cast<float>(src.size().width),
                 static_cast<float>(src.size().height));
  return resize(src, out_size, filter, box);
}

cv::Mat PillowResize::resize(const cv::Mat& src, const cv::Size& out_size, int filter,
                             const cv::Rect2f& box) {
  // Box = x0,y0,w,h
  // Rect = x0,y0,x1,y1
  cv::Vec4f rect(box.x, box.y, box.x + box.width, box.y + box.height);

  int x_size = out_size.width;
  int y_size = out_size.height;
  if (x_size < 1 || y_size < 1) {
    throw std::runtime_error("Height and width must be > 0");
  }

  if (rect[0] < 0.F || rect[1] < 0.F) {
    throw std::runtime_error("Box offset can't be negative");
  }

  if (static_cast<int>(rect[2]) > src.size().width ||
      static_cast<int>(rect[3]) > src.size().height) {
    throw std::runtime_error("Box can't exceed original image size");
  }

  if (box.width < 0 || box.height < 0) {
    throw std::runtime_error("Box can't be empty");
  }

  // If box's coordinates are int and box size matches requested size
  if (static_cast<int>(box.width) == x_size && static_cast<int>(box.height) == y_size) {
    cv::Rect roi = box;
    return cv::Mat(src, roi);
  }
  if (filter == INTERPOLATION_NEAREST) {
    return _nearestResample(src, x_size, y_size, rect);
  }
  std::shared_ptr<Filter> filter_p;

  // Check filter.
  switch (filter) {
    case INTERPOLATION_BOX:
      filter_p = std::make_shared<BoxFilter>(BoxFilter());
      break;
    case INTERPOLATION_BILINEAR:
      filter_p = std::make_shared<BilinearFilter>(BilinearFilter());
      break;
    case INTERPOLATION_HAMMING:
      filter_p = std::make_shared<HammingFilter>(HammingFilter());
      break;
    case INTERPOLATION_BICUBIC:
      filter_p = std::make_shared<BicubicFilter>(BicubicFilter());
      break;
    case INTERPOLATION_LANCZOS:
      filter_p = std::make_shared<LanczosFilter>(LanczosFilter());
      break;
    default:
      throw std::runtime_error("unsupported resampling filter");
  }

  return PillowResize::_resample(src, x_size, y_size, filter_p, rect);
}

cv::Mat PillowResize::_nearestResample(const cv::Mat& im_in, int x_size, int y_size,
                                       const cv::Vec4f& rect) {
  auto x0 = static_cast<int>(rect[0]);
  auto y0 = static_cast<int>(rect[1]);
  auto x1 = static_cast<int>(rect[2]);
  auto y1 = static_cast<int>(rect[3]);
  x0 = std::max(x0, 0);
  y0 = std::max(y0, 0);
  x1 = std::min(x1, im_in.size().width);
  y1 = std::min(y1, im_in.size().height);

  // Affine tranform matrix.
  cv::Mat m = cv::Mat::eye(2, 3, CV_64F);
  m.at<double>(0, 0) = (x1 - x0) / static_cast<double>(x_size);
  m.at<double>(0, 2) = x0;
  m.at<double>(1, 1) = (y1 - y0) / static_cast<double>(y_size);
  m.at<double>(1, 2) = y0;

  cv::Mat im_out = cv::Mat::zeros(y_size, x_size, im_in.type());

  /**
   * \brief affineTransform Transform a point according to the given transformation.
   *
   * \param[in] p Point that has to be transformed.
   * \param[in] a 2×3 transformation matrix.
   *
   * \return Transformed point.
   */
  auto affineTransform = [](const cv::Point& p, const cv::Matx23d& a) -> cv::Point2d {
    const double half_pixel = 0.5;
    double xin = p.x + half_pixel;
    double yin = p.y + half_pixel;

    double xout = a(0, 0) * xin + a(0, 1) * yin + a(0, 2);
    double yout = a(1, 0) * xin + a(1, 1) * yin + a(1, 2);

    return cv::Point2d(xout, yout);
  };

  /**
   * \brief Apply nearest interpolation.
   * Copy a input pixel into the given output matrix position.
   * The copy is performed copying a continuous number of bytes (pixel_size)
   * from input matrix to output matrix.
   *
   * \param[in] im_in Input matrix.
   * \param[out] im_out Output matrix.
   * \param[in] p_in Input point coordinates.
   * \param[in] p_out Output point coordinates.
   * \param[in] pixel_size Size of the pixel in bytes.
   */
  auto nearestInterpolation = [](const cv::Mat& im_in, cv::Mat& im_out, const cv::Point2d& p_in,
                                 const cv::Point& p_out, size_t pixel_size) -> void {
    // Round input coordinates.
    int x = p_in.x < 0 ? -1 : static_cast<int>(p_in.x);
    int y = p_in.y < 0 ? -1 : static_cast<int>(p_in.y);
    if (x < 0 || x >= im_in.size().width || y < 0 || y >= im_in.size().height) {
      return;
    }
    // Copy the input pixel into the output matrix.
    memcpy(im_out.ptr(p_out.y, p_out.x), im_in.ptr(y, x), pixel_size);
  };

  // Check pixel type and determine the pixel size
  // (element size * number of channels).
  size_t pixel_size = 0;
  switch (_getPixelType(im_in)) {
    case CV_8U:
      pixel_size = sizeof(uchar);
      break;
    case CV_8S:
      pixel_size = sizeof(char);
      break;
    case CV_16U:
      pixel_size = sizeof(std::uint16_t);
      break;
    case CV_16S:
      pixel_size = sizeof(std::int16_t);
      break;
    case CV_32S:
      pixel_size = sizeof(int);
      break;
    case CV_32F:
      pixel_size = sizeof(float);
      break;
    default:
      throw std::runtime_error("Pixel type not supported");
  }
  pixel_size *= im_in.channels();

  for (int y = 0; y < im_out.size().height; ++y) {
    for (int x = 0; x < im_out.size().width; ++x) {
      cv::Point out_p(static_cast<int>(x), static_cast<int>(y));
      // Compute input pixel position (corresponding nearest pixel).
      cv::Point2d p = affineTransform(out_p, m);
      // Copy input pixel into output.
      nearestInterpolation(im_in, im_out,
                           cv::Point(static_cast<int>(p.x) - x0, static_cast<int>(p.y) - y0), out_p,
                           pixel_size);
    }
  }
  return im_out;
}

cv::Mat PillowResize::_resample(const cv::Mat& im_in, int x_size, int y_size,
                                const std::shared_ptr<Filter>& filter_p, const cv::Vec4f& rect) {
  cv::Mat im_out;
  cv::Mat im_temp;

  std::vector<int> bounds_horiz;
  std::vector<int> bounds_vert;
  std::vector<double> kk_horiz;
  std::vector<double> kk_vert;

  bool need_horizontal =
      x_size != im_in.size().width || (rect[0] != 0.0F) || static_cast<int>(rect[2]) != x_size;
  bool need_vertical =
      y_size != im_in.size().height || (rect[1] != 0.0F) || static_cast<int>(rect[3]) != y_size;

  // Compute horizontal filter coefficients.
  int ksize_horiz = _precomputeCoeffs(im_in.size().width, rect[0], rect[2], x_size, filter_p,
                                      bounds_horiz, kk_horiz);

  // Compute vertical filter coefficients.
  int ksize_vert = _precomputeCoeffs(im_in.size().height, rect[1], rect[3], y_size, filter_p,
                                     bounds_vert, kk_vert);

  // First used row in the source image.
  int ybox_first = bounds_vert[0];
  // Last used row in the source image.
  int ybox_last = bounds_vert[y_size * 2 - 2] + bounds_vert[y_size * 2 - 1];

  // Two-pass resize, horizontal pass.
  if (need_horizontal) {
    // Shift bounds for vertical pass.
    for (int i = 0; i < y_size; ++i) {
      bounds_vert[i * 2] -= ybox_first;
    }

    // Create destination image with desired ouput width and same input pixel type.
    im_temp.create(ybox_last - ybox_first, x_size, im_in.type());
    if (!im_temp.empty()) {
      _resampleHorizontal(im_temp, im_in, ybox_first, ksize_horiz, bounds_horiz, kk_horiz);
    } else {
      return cv::Mat();
    }
    im_out = im_temp;
  }

  // Vertical pass.
  if (need_vertical) {
    const auto now_w = (im_temp.size().width != 0) ? im_temp.size().width : x_size;
    // Create destination image with desired ouput size and same input pixel type.
    im_out.create(y_size, now_w, im_in.type());
    if (!im_out.empty()) {
      if (im_temp.empty()) {
        im_temp = im_in;
      }
      // Input can be the original image or horizontally resampled one.
      _resampleVertical(im_out, im_temp, ksize_vert, bounds_vert, kk_vert);
    } else {
      return cv::Mat();
    }
  }

  // None of the previous steps are performed, copying.
  if (im_out.empty()) {
    im_out = im_in;
  }

  return im_out;
}

void PillowResize::_resampleHorizontal(cv::Mat& im_out, const cv::Mat& im_in, int offset, int ksize,
                                       const std::vector<int>& bounds,
                                       const std::vector<double>& prekk) {
  // Check pixel type.
  switch (_getPixelType(im_in)) {
    case CV_8U:
      return _resampleHorizontal<uchar>(im_out, im_in, offset, ksize, bounds, prekk,
                                        _normalizeCoeffs8bpc,
                                        static_cast<double>(1U << (precision_bits - 1U)), _clip8);
    case CV_8S:
      return _resampleHorizontal<char>(im_out, im_in, offset, ksize, bounds, prekk, nullptr, 0.,
                                       _roundUp<char>);
    case CV_16U:
      return _resampleHorizontal<std::uint16_t>(im_out, im_in, offset, ksize, bounds, prekk);
    case CV_16S:
      return _resampleHorizontal<std::int16_t>(im_out, im_in, offset, ksize, bounds, prekk, nullptr,
                                               0., _roundUp<std::int16_t>);
    case CV_32S:
      return _resampleHorizontal<int>(im_out, im_in, offset, ksize, bounds, prekk, nullptr, 0.,
                                      _roundUp<int>);
    case CV_32F:
      return _resampleHorizontal<float>(im_out, im_in, offset, ksize, bounds, prekk);
    default:
      throw std::runtime_error("Pixel type not supported");
  }
}

void PillowResize::_resampleVertical(cv::Mat& im_out, const cv::Mat& im_in, int ksize,
                                     const std::vector<int>& bounds,
                                     const std::vector<double>& prekk) {
  im_out = im_out.t();
  _resampleHorizontal(im_out, im_in.t(), 0, ksize, bounds, prekk);
  im_out = im_out.t();
}

#endif