#pragma once
#include <cstddef>
#include <vector>

// #include <torch/torch.h>

namespace kvcache {

struct KVCacheConfig {
  size_t layer_num = 32;
  size_t max_seq_len = 2048;
  size_t hidden_size = 4096;
  size_t num_heads = 32;
  size_t elemsize = 2;
  size_t granularitySize = 2 * 1024 * 1024;
  size_t max_concurrent_requests = 256;
  size_t max_batch_size = 0;

  // schedule
  size_t reserve_prefill{64};
  size_t reserve_decode{1};
  int device_id{-1};
};
struct KVCacheAllocParams {
  std::string request_id;
  size_t kvcache_seq_len;
  int generated_token_len_hit = -1;
  size_t max_new_tokens;
};

struct KVCacheState {
  size_t iter_index;
  size_t seq_len_with_output;
  // size_t prefill_seq_len_with_output;
  int generated_token_len_hit = -1;
  size_t max_new_tokens = 2048;
  // bool cpu_offloaded = false;

  // size_t request_index{0};
};

struct CacheParams {
  std::string request_id;
  char* src_k;
  char* src_v;
  size_t size;
  void* event = nullptr;
};

struct KVCacheTensor {
  std::vector<char*> k;
  std::vector<char*> v;
  size_t curr_layer_index = 0;
  size_t offset = 0;
  size_t step = 0;
  std::vector<int64_t> shape{};
  size_t elemsize = 2;
};

// using StepOutput = std::tuple<std::vector<char*>, std::unordered_set<std::string>>;
// using StepOutput = std::unordered_map<std::string, std::vector<char*>>;
using StepOutput = std::unordered_set<std::string>;
using MemOutput = std::vector<char*>;
using QueryOutput =
    void;  // std::vector<std::vector<torch::Tensor>>;  // layer_num[key_input, value_input,
           //  key_output, value_output]

// using MemOutput = torch::Tensor;
}  // namespace kvcache