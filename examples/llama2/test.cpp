
#include <cstdio>
#include <iostream>
#include <memory>
#include <cstdio>
#include <stdexcept>
#include <cassert>
#include <unordered_set>
#include <mutex>
// #include <lock_guard>
#include "cuda_runtime.h"

#include <cuda.h>

#define LOGE(format, ...)                                          \
  do {                                                             \
    fprintf(stdout, "L%d: " format "\n", __LINE__, ##__VA_ARGS__); \
    fflush(stdout);                                                \
  } while (0)

#define ASSERT(cond, ...)                                   \
  do {                                                      \
    if (!(cond)) {                                          \
      LOGE(__VA_ARGS__);                                    \
      throw std::runtime_error("Assertion failed: " #cond); \
    }                                                       \
  } while (0)

#define DRV_CALL(call)                                                                            \
  do {                                                                                            \
    CUresult result = (call);                                                                     \
    if (CUDA_SUCCESS != result) {                                                                 \
      const char* errMsg;                                                                         \
      cuGetErrorString(result, &errMsg);                                                          \
      ASSERT(0, "Error when exec " #call " %s-%d code:%d err:%s", __FUNCTION__, __LINE__, result, \
             errMsg);                                                                             \
    }                                                                                             \
  } while (0)

#define CUDA_CALL(x)                                                                               \
  do {                                                                                             \
    cudaError_t result = (x);                                                                      \
    if (result != cudaSuccess) {                                                                   \
      const char* errMsg = cudaGetErrorString(result);                                             \
      ASSERT(0, "Error when exec " #x " %s-%d code:%d err:%s", __FUNCTION__, __LINE__, x, errMsg); \
    }                                                                                              \
  } while (0)
CUstream stream_;
cudaStream_t cuda_stream_;
inline void* alloc_pinned(size_t size) {
  void* ptr;
  DRV_CALL(cuMemAllocHost(&ptr, size));
  return ptr;
}

inline void* virtual_alloc(size_t len) {
  CUdeviceptr virtual_ptr;
  DRV_CALL(cuMemAddressReserve(&virtual_ptr, len, 0ULL, 0ULL, 0ULL));

  return (void*)virtual_ptr;
}

void offload_memcpy2d(void* cpu_ptr, void* cuptr, size_t w, size_t h, size_t srcPitch) {
  for (size_t i = 0; i < h; ++i) {
    DRV_CALL(cuMemcpyDtoHAsync(cpu_ptr, (CUdeviceptr)cuptr, w, stream_));
  }
  DRV_CALL(cuStreamSynchronize(stream_));

  return;
  CUDA_MEMCPY2D copy_param = {};
  copy_param.WidthInBytes = w;
  copy_param.Height = h;

  copy_param.dstXInBytes = 0;
  copy_param.dstY = 0;
  copy_param.dstMemoryType = CU_MEMORYTYPE_HOST;
  copy_param.dstHost = cpu_ptr;
  copy_param.dstPitch = w;

  copy_param.srcXInBytes = 0;
  copy_param.srcY = 0;
  copy_param.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copy_param.srcDevice = (CUdeviceptr)cuptr;
  copy_param.srcPitch = srcPitch;
  DRV_CALL(cuMemcpy2DAsync(&copy_param, stream_));
  DRV_CALL(cuStreamSynchronize(stream_));
}

void offload_cudamemcpy2d(void* cpu_ptr, void* cuptr, size_t w, size_t h, size_t srcPitch) {
  CUDA_CALL(cudaMemcpy2D(cpu_ptr, w, cuptr, srcPitch, w, h, cudaMemcpyDeviceToHost));
  // CUDA_CALL(cudaStreamSynchronize(cuda_stream_));
}

class PhyBlock {
 public:
  PhyBlock(int device_id, size_t block_size) : device_id_(device_id), block_size_(block_size) {
    accessDesc_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc_.location.id = device_id_;
    accessDesc_.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  }
  ~PhyBlock() { release(); }
  bool allocate() {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id_;

    auto status = cuMemCreate(&alloc_handle_, block_size_, &prop, 0ULL);
    auto need_release = (CUDA_SUCCESS == status);
    return need_release;
  }

  void map(char* virtual_ptr) {
    assert(virtual_ptr_ == nullptr);
    DRV_CALL(cuMemMap(reinterpret_cast<CUdeviceptr>(virtual_ptr), block_size_, 0ULL, alloc_handle_,
                      0ULL));

    DRV_CALL(
        cuMemSetAccess(reinterpret_cast<CUdeviceptr>(virtual_ptr), block_size_, &accessDesc_, 1));
    virtual_ptr_ = virtual_ptr;
  }

  void unmap(char* virtual_ptr) {
    // todo : multiple map
    if (virtual_ptr_ == nullptr) {
      return;
    }
    DRV_CALL(cuMemUnmap(reinterpret_cast<CUdeviceptr>(virtual_ptr_), block_size_));
    virtual_ptr_ = nullptr;
  }

  void release() {
    unmap((char*)virtual_ptr_);

    if (need_release_) cuMemRelease(alloc_handle_);
    need_release_ = false;
  }

 private:
  int device_id_{-1};
  size_t block_size_;
  CUmemGenericAllocationHandle alloc_handle_;
  char* virtual_ptr_{nullptr};

  CUmemAccessDesc accessDesc_ = {};
  bool need_release_{false};
};

int main() {
  int device_id = 0;
  // IPIPE_ASSERT(device_id >= 0);
  cudaSetDevice(device_id);
  cudaFree(0);

  int deviceCount = 0;
  DRV_CALL(cuDeviceGetCount(&deviceCount));

  CUdevice cu_dev;
  DRV_CALL(cuDeviceGet(&cu_dev, device_id));

  CUcontext cu_ctx;
  DRV_CALL(cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev));
  DRV_CALL(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING));
  CUDA_CALL(cudaStreamCreate(&cuda_stream_));

  int seq_len = 15;
  int need_blk = seq_len / 256;
  if (seq_len % 256 != 0) {
    need_blk++;
  }
  int hidden_size = 4096;
  void* cpu_ptr = alloc_pinned(seq_len * 2 * hidden_size * 32);
  int context_max_size_ = 2 * 2048 * hidden_size;
  char* cuptr = (char*)virtual_alloc(context_max_size_ * 32);
  char* ori_cu = cuptr;
  int w = seq_len * 2 * hidden_size;
  int h = 32;
  size_t srcPitch = 2 * 2048 * hidden_size;

  for (int i = 0; i < 32; ++i) {
    char* local_ptr = cuptr;
    for (int j = 0; j < need_blk; ++j) {
      auto* blk = new PhyBlock(0, 2 * 1024 * 1024);
      assert(blk->allocate());
      blk->map((char*)local_ptr);
      local_ptr += 2 * 1024 * 1024;
    }
    cuptr += srcPitch;
    if (need_blk == 8 && local_ptr != cuptr)
      throw std::runtime_error("Assertion failed: local_ptr != cuptr");
  }

  // cudaMalloc(&ori_cu, context_max_size_ * 32);

  std::cout << (void*)ori_cu << std::endl;
  // offload_cudamemcpy2d(cpu_ptr, ori_cu, w, h, srcPitch);
  offload_memcpy2d(cpu_ptr, ori_cu, w, h, srcPitch);

  return 0;
}