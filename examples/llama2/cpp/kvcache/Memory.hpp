#pragma once

#include <cstdio>

#include <memory>
#include <cstdio>
#include <stdexcept>
#include <cassert>
#include <unordered_set>
#include <mutex>
// #include <lock_guard>

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

namespace kvcache {
class PhyBlock {
 public:
  PhyBlock(int device_id, size_t block_size);
  ~PhyBlock();
  bool allocate() {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id_;

    auto status = cuMemCreate(&alloc_handle_, block_size_, &prop, 0ULL);
    auto need_release = (CUDA_SUCCESS == status);
    return need_release;
  }

  void map(char* virtual_ptr);

  void unmap(char* virtual_ptr);

  void release() {
    unmap(virtual_ptr_);

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

class PyhBlkPool {
 public:
  PyhBlkPool(int device_id, size_t block_size);

  ~PyhBlkPool() = default;
  size_t size() { return phy_blocks_.size(); }
  size_t get_system_free_memory();
  size_t query_system_free_memory(double factor);
  size_t alloc(size_t num_blocks) {
    size_t num_alloc = 0;
    for (; num_alloc < num_blocks; num_alloc++) {
      auto pyh = std::make_shared<PhyBlock>(device_id_, block_size_);
      if (!pyh->allocate()) {
        num_allocated_ += num_alloc;
        return num_alloc;
      }
      phy_blocks_.push(pyh);
    }
    num_allocated_ += num_alloc;
    return num_blocks;
  }

  // CUresult cuMemcpy2DAsync	(	const CUDA_MEMCPY2D * 	pCopy,
  // CUstream 	hStream
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

  void onload_memcpy2d(void* cuptr, void* cpu_ptr, size_t w, size_t h, size_t dstPitch) {
    for (size_t i = 0; i < h; ++i) {
      DRV_CALL(cuMemcpyHtoDAsync((CUdeviceptr)cuptr, cpu_ptr, w, stream_));
    }
    DRV_CALL(cuStreamSynchronize(stream_));
    return;

    CUDA_MEMCPY2D copy_param = {};
    copy_param.WidthInBytes = w;
    copy_param.Height = h;

    copy_param.srcXInBytes = 0;
    copy_param.srcY = 0;
    copy_param.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy_param.srcHost = cpu_ptr;
    copy_param.srcPitch = w;

    copy_param.dstXInBytes = 0;
    copy_param.dstY = 0;
    copy_param.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy_param.dstDevice = (CUdeviceptr)cuptr;
    copy_param.dstPitch = dstPitch;
    DRV_CALL(cuMemcpy2DAsync(&copy_param, stream_));
    DRV_CALL(cuStreamSynchronize(stream_));
  }

  std::shared_ptr<PhyBlock> get_free_blk() {
    if (phy_blocks_.empty()) {
      return nullptr;
    }

    auto re = phy_blocks_.front();
    phy_blocks_.pop();
    return re;
  }

  void free(std::shared_ptr<PhyBlock> pyh) { phy_blocks_.push(pyh); }

  size_t free_reserved(ssize_t num_blocks) {
    size_t num_freed = 0;
    for (; num_freed < num_blocks; num_freed++) {
      if (!reserved_phy_blocks_.empty()) {
        auto pyh = reserved_phy_blocks_.front();
        reserved_phy_blocks_.pop();
        phy_blocks_.push(pyh);
      } else {
        break;
      }
    }

    return num_freed;
  }

  void reserve(std::shared_ptr<PhyBlock> pyh) { reserved_phy_blocks_.push(pyh); }
  std::shared_ptr<PhyBlock> get_reserve_blk() {
    if (reserved_phy_blocks_.empty()) {
      return nullptr;
    }

    auto re = reserved_phy_blocks_.front();
    reserved_phy_blocks_.pop();
    return re;
  }

  // std::unordered_set<std::shared_ptr<PhyBlock>> get_phy_blocks(size_t num_blocks) {
  //   std::unordered_set<std::shared_ptr<PhyBlock>> blocks;
  //   std::lock_guard<std::mutex> lock(blk_mtx_);
  //   for (size_t i = 0; i < num_blocks; i++) {
  //     // if (phy_blocks_.empty()) {
  //     //   return blocks;
  //     // }
  //     blocks.insert(phy_blocks_.front());
  //     used_phy_blocks_.insert(phy_blocks_.front());
  //     phy_blocks_.pop();
  //   }
  //   return blocks;
  // }

  // void release_phy_blocks(const std::vector<std::shared_ptr<PhyBlock>>& blocks) {
  //   // std::lock_guard<std::mutex> lock(blk_mtx_);
  //   // for (const auto& blk : blocks) {
  //   //   used_phy_blocks_.erase(blk);
  //   //   phy_blocks_.push(blk);
  //   // }
  // }

 private:
  CUstream stream_;
  std::mutex blk_mtx_;
  std::queue<std::shared_ptr<PhyBlock>> phy_blocks_;
  // std::unordered_set<std::shared_ptr<PhyBlock>> used_phy_blocks_;
  std::queue<std::shared_ptr<PhyBlock>> reserved_phy_blocks_;

  int device_id_;
  size_t block_size_;
  // size_t num_blocks;
  size_t system_mem_{0};

  size_t num_allocated_{0};
};

inline void* virtual_alloc(size_t len) {
  CUdeviceptr virtual_ptr;
  DRV_CALL(cuMemAddressReserve(&virtual_ptr, len, 0ULL, 0ULL, 0ULL));

  return (void*)virtual_ptr;
}

inline void virtual_free(void* ptr, size_t len) {
  CUdeviceptr virtual_ptr = (CUdeviceptr)ptr;
  DRV_CALL(cuMemAddressFree(virtual_ptr, len));
}

// bool map_virtual_to_pyh(char* virtual_ptr, PhyBlock* ptr_blk, size_t granularitySize) {
//   CUcontext currentContext;
//   CUresult status = cuCtxGetCurrent(&currentContext);
//   ptr_blk->map(virtual_ptr);
//   status =
//       cuMemMap(reinterpret_cast<CUdeviceptr>(virtual_ptr), granularitySize, 0ULL, phy_handle,
//       0ULL);
//   if (status != CUDA_SUCCESS) {
//     return false;
//   }
//   return true;
// }

inline void* alloc_pinned(size_t size) {
  void* ptr;
  DRV_CALL(cuMemAllocHost(&ptr, size));
  return ptr;
}

void init_device(int device_id);
inline void free_pinned(void* ptr) { DRV_CALL(cuMemFreeHost(ptr)); }

}  // namespace kvcache