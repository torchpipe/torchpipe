/* Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// modified from
// https://github.com/CVCUDA/CV-CUDA/blob/release_v0.2.x/src/cvcuda/priv/legacy/pillow_resize.cu
// #include <torch/torch.h>

#include "cvcuda/PillowResizeCudaV2.hpp"
#include <cuda_runtime.h>

#define BLOCK 32
#define SHARE_MEM_LIMIT 4096
#include "at_replace.cuh"
namespace ipipe_nvcv {

// Used to disambiguate between the constructors that accept legacy memory buffers,
// and the ones that accept the new ones. Just pass NewAPI as first parameter.
struct NewAPITag {};

constexpr NewAPITag NewAPI = {};

template <typename T>
struct Ptr2dNHWC {
  typedef T value_type;

  __host__ __device__ __forceinline__ Ptr2dNHWC()
      : batches(0), rows(0), cols(0), imgStride(0), rowStride(0), ch(0) {}

  __host__ __device__ __forceinline__ Ptr2dNHWC(int rows_, int cols_, int ch_, T *data_)
      : batches(1),
        rows(rows_),
        cols(cols_),
        ch(ch_),
        imgStride(0),
        rowStride(CalcNHWCRowStride<T>(cols_, ch_)),
        data(data_) {}

  __host__ __device__ __forceinline__ Ptr2dNHWC(int batches_, int rows_, int cols_, int ch_,
                                                T *data_)
      : batches(batches_),
        rows(rows_),
        cols(cols_),
        ch(ch_),
        imgStride(CalcNHWCImageStride<T>(rows_, cols_, ch_)),
        rowStride(CalcNHWCRowStride<T>(cols_, ch_)),
        data(data_) {}

  __host__ __device__ __forceinline__ Ptr2dNHWC(NewAPITag, int rows_, int cols_, int ch_,
                                                int rowStride_, T *data_)
      : batches(1),
        rows(rows_),
        cols(cols_),
        ch(ch_),
        imgStride(0),
        rowStride(rowStride_),
        data(data_) {}

  __host__ __forceinline__ Ptr2dNHWC(const TensorDataAccessStridedImagePlanar &tensor)
      : batches(tensor.size(0)),
        rows(tensor.size(1)),
        cols(tensor.size(2)),
        ch(tensor.size(3)),
        imgStride(CalcNHWCImageStride<T>(rows, cols, ch)),
        rowStride(CalcNHWCRowStride<T>(cols, ch)),
        data(tensor.data_ptr<T>()) {}

  // ptr for uchar1/3/4, ushort1/3/4, float1/3/4, typename T -> uchar3 etc.
  // each fetch operation get a x-channel elements
  __host__ __device__ __forceinline__ T *ptr(int b, int y, int x) {
    // return (T *)(data + b * rows * cols + y * cols + x);
    return (T *)(reinterpret_cast<unsigned char *>(data) + b * imgStride + y * rowStride +
                 x * sizeof(T));
  }

  const __host__ __device__ __forceinline__ T *ptr(int b, int y, int x) const {
    // return (const T *)(data + b * rows * cols + y * cols + x);
    return (const T *)(reinterpret_cast<const unsigned char *>(data) + b * imgStride +
                       y * rowStride + x * sizeof(T));
  }

  // ptr for uchar, ushort, float, typename T -> uchar etc.
  // each fetch operation get a single channel element
  __host__ __device__ __forceinline__ T *ptr(int b, int y, int x, int c) {
    // return (T *)(data + b * rows * cols * ch + y * cols * ch + x * ch + c);
    return (T *)(reinterpret_cast<unsigned char *>(data) + b * imgStride + y * rowStride +
                 (x * ch + c) * sizeof(T));
  }

  const __host__ __device__ __forceinline__ T *ptr(int b, int y, int x, int c) const {
    // return (const T *)(data + b * rows * cols * ch + y * cols * ch + x * ch + c);
    return (const T *)(reinterpret_cast<const unsigned char *>(data) + b * imgStride +
                       y * rowStride + (x * ch + c) * sizeof(T));
  }

  __host__ __device__ __forceinline__ int at_rows(int b) { return rows; }

  __host__ __device__ __forceinline__ int at_rows(int b) const { return rows; }

  __host__ __device__ __forceinline__ int at_cols(int b) { return cols; }

  __host__ __device__ __forceinline__ int at_cols(int b) const { return cols; }

  int batches;
  int rows;
  int cols;
  int ch;
  int imgStride;
  int rowStride;
  T *data;
};

class BilinearFilter {
 public:
  __host__ __device__ BilinearFilter() : _support(bilinear_filter_support) {};

  __host__ __device__ work_type filter(work_type x) {
    if (x < 0.0) {
      x = -x;
    }
    if (x < 1.0) {
      return 1.0 - x;
    }
    return 0.0;
  }

  __host__ __device__ work_type support() const { return _support; };

 private:
  work_type _support;
};

template <class Filter>
__global__ void _precomputeCoeffs(int in_size, int in0, work_type scale, work_type filterscale,
                                  work_type support, int out_size, int k_size, Filter filterp,
                                  int *bounds_out, work_type *kk_out, bool normalize_coeff,
                                  bool use_share_mem) {
  const int xx = blockIdx.x * blockDim.x + threadIdx.x;
  const int local_id = threadIdx.x;
  const int x_offset = blockIdx.x * blockDim.x;
  work_type *kk = kk_out + x_offset * k_size;
  if (use_share_mem) {
    extern __shared__ __align__(sizeof(work_type)) unsigned char smem_raw[];
    kk = reinterpret_cast<work_type *>(smem_raw);
  }

  if (xx < out_size) {
    int x = 0;
    int xmin = 0;
    int xmax = 0;
    work_type center = 0;
    work_type ww = 0;
    work_type ss = 0;

    const work_type half_pixel = 0.5;

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
    work_type *k = &kk[local_id * k_size];
    for (x = 0; x < xmax; ++x) {
      work_type w = filterp.filter((x + xmin - center + half_pixel) * ss);
      k[x] = w;
      ww += w;
    }
    for (x = 0; x < xmax; ++x) {
      if (std::fabs(ww) > 1e-5) {
        k[x] /= ww;
      }
    }
    // Remaining values should stay empty if they are used despite of xmax.
    for (; x < k_size; ++x) {
      k[x] = .0f;
    }
    if (normalize_coeff) {
      for (int i = 0; i < k_size; i++) {
        work_type val = k[i];
        if (val < 0) {
          k[i] = static_cast<int>(-half_pixel + val * (1U << precision_bits));
        } else {
          k[i] = static_cast<int>(half_pixel + val * (1U << precision_bits));
        }
      }
    }

    bounds_out[xx * 2] = xmin;
    bounds_out[xx * 2 + 1] = xmax;
  }
  if (use_share_mem) {
    __syncthreads();
    for (int i = local_id; i < (out_size - x_offset) * k_size && i < blockDim.x * k_size;
         i += blockDim.x) {
      kk_out[x_offset * k_size + i] = kk[i];
    }
  }
}

template <class T, class Filter>
__global__ void horizontal_pass(const Ptr2dNHWC<T> src, Ptr2dNHWC<T> dst, NVCVRectI roi,
                                Filter &filterp, int h_ksize, int v_ksize, int *h_bounds,
                                work_type *h_kk, int *v_bounds, work_type *v_kk,
                                work_type init_buffer, bool round_up, bool use_share_mem) {
  const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int x_offset = blockIdx.x * blockDim.x;
  const int batch_idx = get_batch_idx();
  int out_height = dst.rows, out_width = dst.cols;
  work_type *h_k_tmp = h_kk + x_offset * h_ksize;

  if (use_share_mem) {
    const int local_tid = threadIdx.x + blockDim.x * threadIdx.y;
    extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_h[];
    h_k_tmp = reinterpret_cast<work_type *>(kk_smem_h);

    for (int i = local_tid; i < blockDim.x * h_ksize && i < (out_width - x_offset) * h_ksize;
         i += blockDim.x * blockDim.y) {
      h_k_tmp[i] = h_kk[x_offset * h_ksize + i];
    }

    __syncthreads();
  }

  if (dst_x < out_width && dst_y < out_height) {
    int xmin = h_bounds[dst_x * 2];
    int xmax = h_bounds[dst_x * 2 + 1];

    work_type *h_k = &h_k_tmp[local_x * h_ksize];

    for (int c = 0; c < src.ch; ++c) {
      work_type h_ss = const_init_buffer;
      for (int x = 0; x < xmax; ++x) {
        h_ss = h_ss + *src.ptr(batch_idx, dst_y, x + xmin, c) * h_k[x];
      }
      // printf("\nh_ss %f \n", h_ss);

      *dst.ptr(batch_idx, dst_y, dst_x, c) = SaturateCastPillow(h_ss);
    }
  }
}

template <class T, class Filter>
__global__ void vertical_pass(const Ptr2dNHWC<T> src, Ptr2dNHWC<T> dst, NVCVRectI roi,
                              Filter &filterp, int h_ksize, int v_ksize, int *h_bounds,
                              work_type *h_kk, int *v_bounds, work_type *v_kk,
                              work_type init_buffer, bool round_up, bool use_share_mem) {
  const int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_y = threadIdx.y;
  const int y_offset = blockIdx.y * blockDim.y;
  const int batch_idx = get_batch_idx();
  int out_height = dst.rows, out_width = dst.cols;
  work_type *v_k_tmp = v_kk + y_offset * v_ksize;

  if (use_share_mem) {
    const int local_tid = threadIdx.x + blockDim.x * threadIdx.y;
    extern __shared__ __align__(sizeof(work_type)) unsigned char kk_smem_v[];
    v_k_tmp = reinterpret_cast<work_type *>(kk_smem_v);

    for (int i = local_tid; i < blockDim.y * v_ksize && i < (out_height - y_offset) * v_ksize;
         i += blockDim.x * blockDim.y) {
      v_k_tmp[i] = v_kk[y_offset * v_ksize + i];
    }

    __syncthreads();
  }

  if (dst_x < out_width && dst_y < out_height) {
    int ymin = v_bounds[dst_y * 2];
    int ymax = v_bounds[dst_y * 2 + 1];

    work_type *v_k = &v_k_tmp[local_y * v_ksize];

    for (int c = 0; c < src.ch; ++c) {
      work_type ss = const_init_buffer;
      for (int y = 0; y < ymax; ++y) {
        ss = ss + *src.ptr(batch_idx, y + ymin, dst_x, c) * v_k[y];
      }

      *dst.ptr(batch_idx, dst_y, dst_x, c) = SaturateCastPillow(ss);
    }
  }
}

template <typename Filter, typename elem_type>
void pillow_resize_v2(const TensorDataAccessStridedImagePlanar &inData,
                      const TensorDataAccessStridedImagePlanar &outData, void *gpu_workspace,
                      bool normalize_coeff, work_type init_buffer, bool round_up,
                      cudaStream_t stream) {
  // auto input_shape = inData.size();
  int input_shape_N = inData.size(0);
  auto input_shape_H = inData.size(1);
  auto input_shape_W = inData.size(2);
  auto input_shape_C = inData.size(3);
  Ptr2dNHWC<elem_type> src_ptr(inData);
  Ptr2dNHWC<elem_type> dst_ptr(outData);
  NVCVRectI roi = {0, 0, src_ptr.cols, src_ptr.rows};
  Filter filterp;
  work_type h_scale = 0, v_scale = 0;
  work_type h_filterscale = 0, v_filterscale = 0;
  h_filterscale = h_scale = static_cast<work_type>(roi.width) / dst_ptr.cols;
  v_filterscale = v_scale = static_cast<work_type>(roi.height) / dst_ptr.rows;

  int out_width = dst_ptr.cols;
  int out_height = dst_ptr.rows;
  assert((roi.x == 0) && roi.width == input_shape_W);
  const bool need_horizontal = out_width != input_shape_W;
  const bool need_vertical = out_height != input_shape_H;

  if (!need_horizontal && !need_vertical) {
    assert(false);
    throw std::runtime_error("no need resize");
  }
  assert((roi.y == 0) && roi.height == input_shape_H);
  // out_height != input_shape_H || (roi.y != 0) || roi.height != out_height;

  if (h_filterscale < 1.0) {
    h_filterscale = 1.0;
  }
  if (v_filterscale < 1.0) {
    v_filterscale = 1.0;
  }

  // Determine support size (length of resampling filter).
  work_type h_support = filterp.support() * h_filterscale;
  work_type v_support = filterp.support() * v_filterscale;

  // Maximum number of coeffs.
  int h_k_size = static_cast<int>(ceil(h_support)) * 2 + 1;
  int v_k_size = static_cast<int>(ceil(v_support)) * 2 + 1;

  work_type *h_kk = (work_type *)((char *)gpu_workspace);
  work_type *v_kk = (work_type *)((char *)h_kk + dst_ptr.cols * h_k_size * sizeof(work_type));
  int *h_bounds = (int *)((char *)v_kk + dst_ptr.rows * v_k_size * sizeof(work_type));
  int *v_bounds = (int *)((char *)h_bounds + dst_ptr.cols * 2 * sizeof(int));
  elem_type *d_h_data = (elem_type *)((char *)v_bounds + dst_ptr.rows * 2 * sizeof(int));

  Ptr2dNHWC<elem_type> h_ptr(input_shape_N, input_shape_H, out_width, input_shape_C,
                             (elem_type *)d_h_data);

  dim3 blockSize(BLOCK, BLOCK / 4, 1);
  dim3 gridSizeH(divUp(out_width, blockSize.x), divUp(input_shape_H, blockSize.y), input_shape_N);
  dim3 gridSizeV(divUp(out_width, blockSize.x), divUp(out_height, blockSize.y), input_shape_N);

  dim3 coef_block(BLOCK * 2, 1, 1);
  dim3 h_coef_grid(divUp(dst_ptr.cols, coef_block.x), 1, 1);
  dim3 v_coef_grid(divUp(dst_ptr.rows, coef_block.x), 1, 1);

  size_t h_sm_size = coef_block.x * (h_k_size * sizeof(work_type));
  size_t v_sm_size = coef_block.x * (v_k_size * sizeof(work_type));

  size_t hv_sm_size1 = h_k_size * sizeof(work_type) * blockSize.x;
  size_t hv_sm_size2 = v_k_size * sizeof(work_type) * blockSize.y;
  bool h_use_share_mem = h_sm_size <= SHARE_MEM_LIMIT;
  if (!h_use_share_mem) {
    h_sm_size = 0;
  }
  bool v_use_share_mem = v_sm_size <= SHARE_MEM_LIMIT;
  if (!v_use_share_mem) {
    v_sm_size = 0;
  }
  bool hv_use_share_mem = (hv_sm_size1 <= SHARE_MEM_LIMIT) && (hv_sm_size2 <= SHARE_MEM_LIMIT);
  if (!hv_use_share_mem) {
    hv_sm_size1 = 0;
    hv_sm_size2 = 0;
  }
  if (need_horizontal)
    // compute horizental coef
    _precomputeCoeffs<Filter><<<h_coef_grid, coef_block, h_sm_size, stream>>>(
        src_ptr.cols, roi.x, h_scale, h_filterscale, h_support, dst_ptr.cols, h_k_size, filterp,
        h_bounds, h_kk, normalize_coeff, h_use_share_mem);

  // checkKernelErrors();
  // #ifdef CUDA_DEBUG_LOG
  //   checkCudaErrors(cudaStreamSynchronize(stream));
  //   checkCudaErrors(cudaGetLastError());
  // #endif

  if (need_vertical)
    // compute vertical coef
    _precomputeCoeffs<Filter><<<v_coef_grid, coef_block, v_sm_size, stream>>>(
        src_ptr.rows, roi.y, v_scale, v_filterscale, v_support, dst_ptr.rows, v_k_size, filterp,
        v_bounds, v_kk, normalize_coeff, v_use_share_mem);

  // checkKernelErrors();
  // #ifdef CUDA_DEBUG_LOG
  //   checkCudaErrors(cudaStreamSynchronize(stream));
  //   checkCudaErrors(cudaGetLastError());
  // #endif
  if (need_horizontal) {
    if (need_vertical) {
      horizontal_pass<elem_type, Filter><<<gridSizeH, blockSize, hv_sm_size1, stream>>>(
          src_ptr, h_ptr, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk,
          init_buffer, round_up, hv_use_share_mem);
    } else {
      horizontal_pass<elem_type, Filter><<<gridSizeH, blockSize, hv_sm_size1, stream>>>(
          src_ptr, dst_ptr, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk,
          init_buffer, round_up, hv_use_share_mem);
    }
  }
  if (need_vertical) {
    if (need_horizontal) {
      vertical_pass<elem_type, Filter><<<gridSizeV, blockSize, hv_sm_size2, stream>>>(
          h_ptr, dst_ptr, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk,
          init_buffer, round_up, hv_use_share_mem);
    } else {
      vertical_pass<elem_type, Filter><<<gridSizeV, blockSize, hv_sm_size2, stream>>>(
          src_ptr, dst_ptr, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk,
          init_buffer, round_up, hv_use_share_mem);
    }
  }

  // checkKernelErrors();
  // vertical_pass<elem_type, Filter><<<gridSizeV, blockSize, hv_sm_size2, stream>>>(
  //     h_ptr, dst_ptr, roi, filterp, h_k_size, v_k_size, h_bounds, h_kk, v_bounds, v_kk,
  //     init_buffer, round_up, hv_use_share_mem);

  // checkKernelErrors();
  // #ifdef CUDA_DEBUG_LOG
  //   checkCudaErrors(cudaStreamSynchronize(stream));
  //   checkCudaErrors(cudaGetLastError());
  // #endif
}

template <typename Filter>
void pillow_resize_filter(const TensorDataAccessStridedImagePlanar &inData,
                          const TensorDataAccessStridedImagePlanar &outData, void *gpu_workspace,
                          cudaStream_t stream) {
  auto data_type = inData.scalar_type();
  switch (data_type) {
    case torch::kByte:
      pillow_resize_v2<Filter, unsigned char>(inData, outData, gpu_workspace, true, 0., false,
                                              stream);
      break;
    // case torch::kChar:
    //   pillow_resize_v2<Filter, signed char>(inData, outData, gpu_workspace, true, 0., false,
    //                                         stream);
    //   break;

    // case torch::kFloat:
    //   pillow_resize_v2<Filter, float>(inData, outData, gpu_workspace, true, 0., false, stream);
    //   break;
    default:
      throw std::runtime_error("scalar_type not supported.");
  }
}
void PillowResizeCudaV2::forward_impl(torch::Tensor data, torch::Tensor out) {
  pillow_resize_filter<BilinearFilter>(data, out, gpu_workspace_,
                                       c10::cuda::getCurrentCUDAStream());
}

}  // namespace ipipe_nvcv
