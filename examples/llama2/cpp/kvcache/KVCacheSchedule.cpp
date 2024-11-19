#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <vector>
#include <unordered_set>
#include <string>
#include <tuple>
#include "KVCacheSchedule.hpp"
#include "ipipe_common.hpp"
#include "base_logging.hpp"
#include "time_utils.hpp"
#include "KVCacheTensorType.hpp"

#include "threadsafe_kv_storage.hpp"
namespace kvcache {

torch::TensorOptions get_tensor_options(size_t elemsize) {
  auto type = torch::kByte;
  switch (elemsize) {
    case 1:
      break;
    case 2:
      type = torch::kFloat16;
      break;
    case 4:
      type = torch::kFloat;
      break;
    default:
      throw std::runtime_error("Unsupported element size(should be 1 ,2 ,4)");
  };

  auto options = torch::TensorOptions()
                     .device(torch::kCUDA, -1)
                     .dtype(type)
                     .layout(torch::kStrided)
                     .requires_grad(false);
  return options;
}

void KVCacheSchedule::init(const KVCacheConfig& config) {
  // Implementation of the init function
  // Initialize the KVCacheSchedule
  config_ = config;
  seq_per_block_ = (config.granularitySize / (config.hidden_size * config.elemsize));
  if (config_.reserve_decode < 1) {
    throw std::runtime_error("reserve_decode should >= 1, but got " +
                             std::to_string(config_.reserve_decode));
  }

  min_blk_ = (2 + 2 * config_.max_seq_len / seq_per_block_) * config_.layer_num;
  memory_ = std::make_unique<KVCacheMemory>();
  memory_->init(config);

  ipipe::TimeGuard guard("KVCacheSchedule: alloc_blocks_up_to");
  size_t re = memory_->alloc_blocks_up_to(min_blk_);
  guard.release();
  SPDLOG_INFO("KVCacheSchedule: try to alloc: {}, get {} blks", min_blk_, re);
  IPIPE_ASSERT(re >= min_blk_);
}

void KVCacheSchedule::alloc_reqid(const KVCacheAllocParams& data) {
  IPIPE_ASSERT(prefill_kvcache_.find(data.request_id) == prefill_kvcache_.end());
  prefill_kvcache_.insert(
      {data.request_id,
       KVCacheState({0, data.kvcache_seq_len, data.generated_token_len_hit, data.max_new_tokens})});
}

void KVCacheSchedule::free_reqid(const std::string& request_id) {
  SPDLOG_INFO("KVCacheSchedule: free_reqid: {}", request_id);
  decode_kvcache_.erase(request_id);
  prefill_kvcache_.erase(request_id);
  memory_->free_reqid(request_id);
}

std::unordered_set<std::string> KVCacheSchedule::valid_prefill_requests(
    size_t valid_blocks, size_t max_batch_size, size_t max_concurrent_requests) {
  std::unordered_set<std::string> valid_reqs;

  size_t need_blk = 0;
  size_t bs = 0;
  for (const auto& item : prefill_kvcache_) {
    size_t target = item.second.seq_len_with_output +
                    std::max(item.second.generated_token_len_hit - 1, int(config_.reserve_prefill));
    target = std::min(target, item.second.seq_len_with_output - 1 + item.second.max_new_tokens);
    target = std::min(target, config_.max_seq_len);
    need_blk += cal_prefill_blocks(target, seq_per_block_, config_.layer_num);

    bs += item.second.seq_len_with_output - 1;

    if (need_blk > valid_blocks || bs > max_batch_size ||
        valid_reqs.size() > max_concurrent_requests) {
      break;
    }
    valid_reqs.insert(item.first);
  }

  return valid_reqs;
}

std::unordered_set<std::string> KVCacheSchedule::cal_offload2cpu(
    const std::unordered_set<std::string>& valid_reqs, size_t blk_size) {
  std::unordered_set<std::string> offload2cpu_blks;

  std::vector<std::string> unvalid_reqs;
  std::vector<size_t> unvalid_reqs_need_blks;
  std::vector<size_t> unvalid_reqs_occupy;

  for (const auto& item : decode_kvcache_) {
    if (valid_reqs.count(item.first)) {
      continue;
    }
    unvalid_reqs.push_back(item.first);
    unvalid_reqs_need_blks.push_back(
        cal_decode_blocks(item.second.seq_len_with_output + config_.reserve_decode, seq_per_block_,
                          config_.layer_num));
    unvalid_reqs_occupy.push_back(
        cal_prefill_blocks(item.second.seq_len_with_output - 1, seq_per_block_, config_.layer_num));
  }
  if (unvalid_reqs.size() == 1) return {unvalid_reqs[0]};
  std::partial_sum(unvalid_reqs_need_blks.begin(), unvalid_reqs_need_blks.end(),
                   unvalid_reqs_need_blks.begin());
  std::partial_sum(unvalid_reqs_occupy.rbegin(), unvalid_reqs_occupy.rend(),
                   unvalid_reqs_occupy.rbegin());
  if (unvalid_reqs_need_blks.back() <= blk_size) {
    return {};
  }
  for (size_t i = unvalid_reqs.size() - 1; i > 0; i--) {
    if (unvalid_reqs_need_blks[i - 1] <= unvalid_reqs_occupy[i] + blk_size) {
      offload2cpu_blks.insert(unvalid_reqs[i]);
    }
  }
  IPIPE_ASSERT(!offload2cpu_blks.empty());
  return offload2cpu_blks;
}

void KVCacheSchedule::prepare_cpu_offload(KVCacheMemory::CachedMemoryTasks& tasks) {
  if (!onload2gpu_kvcache_.empty()) {
    // 此时已经执行完毕， 可用
    decode_kvcache_.insert(onload2gpu_kvcache_.begin(), onload2gpu_kvcache_.end());
    onload2gpu_kvcache_.clear();
  }

  int sum_decode_blks = 0;
  for (const auto& item : decode_kvcache_) {
    sum_decode_blks +=
        (cal_decode_blocks(item.second.seq_len_with_output + 1, seq_per_block_, config_.layer_num));
  }

  // 检查是否需要提前进行offload
  int extra_memory_need = sum_decode_blks - int(memory_->get_free_blocks() +
                                                memory_->get_reserved_blocks() + system_blocks_);
  if (extra_memory_need > 0) {
    std::unordered_set<std::string> offload2cpu_req;
    for (auto iter = decode_kvcache_.rbegin(); iter != decode_kvcache_.rend(); iter++) {
      offload2cpu_req.insert(iter->first);
      extra_memory_need -= cal_prefill_blocks(iter->second.seq_len_with_output + 1, seq_per_block_,
                                              config_.layer_num);

      if (extra_memory_need <= 0) break;
    }
    tasks.offload2cpu = offload2cpu_req;
    for (const auto& item : offload2cpu_req) {
      auto iter = decode_kvcache_.find(item);
      offload2cpu_kvcache_.insert(*iter);
      decode_kvcache_.erase(iter);
    }
  } else {
    int sum_offload2cpu_blks = 0;
    // todo : 实时分次的把数据转移到cpu
    for (const auto& item : offload2cpu_kvcache_) {
      sum_offload2cpu_blks += cal_prefill_blocks(item.second.seq_len_with_output + 1,
                                                 seq_per_block_, config_.layer_num);
    }
    if (sum_offload2cpu_blks > 0 && (sum_offload2cpu_blks + sum_decode_blks <=
                                     memory_->get_free_blocks() + memory_->get_reserved_blocks())) {
      // 检查将cpu上的数据转移回GPU的时机
      // system blks 可能还没有申请完成， reserved blks 可能还没有释放完成
      for (const auto& kv : offload2cpu_kvcache_) {
        tasks.onload2gpu.insert(kv.first);
      }
      onload2gpu_kvcache_.insert(offload2cpu_kvcache_.begin(), offload2cpu_kvcache_.end());
      offload2cpu_kvcache_.clear();
    }
  }
}

void KVCacheSchedule::prepare_next_memory(KVCacheMemory::CachedMemoryTasks& tasks) {
  // if (!memory_->block_allocable()) return;
  // if (get_system_blocks() + get_reserved_blocks() <= 0) return;
  // int need_alloc_blk = 0;

  // system blocks
  // std::vector<size_t> new_blk_next_step;
  int sum_decode_blks = 0, sum_prefill_blks = 0, sum_offload2cpu_blks = 0;
  for (const auto& item : decode_kvcache_) {
    sum_decode_blks +=
        (cal_decode_blocks(item.second.seq_len_with_output + 1, seq_per_block_, config_.layer_num));
  }
  for (const auto& item : prefill_kvcache_) {
    sum_prefill_blks += (cal_prefill_blocks(item.second.seq_len_with_output + 1, seq_per_block_,
                                            config_.layer_num));
  }
  for (const auto& item : offload2cpu_kvcache_) {
    sum_offload2cpu_blks +=
        cal_prefill_blocks(item.second.seq_len_with_output + 1, seq_per_block_, config_.layer_num);
  }
  // decode优先使用系统blk
  int need_alloc_blk = sum_decode_blks - memory_->get_free_blocks();

  // int(get_free_blocks() + get_system_blocks()) - sum_decode_blks;

  // 显存不足，必须从系统申请
  need_alloc_blk = std::max(need_alloc_blk,
                            int(sum_decode_blks + sum_prefill_blks + sum_offload2cpu_blks) -
                                int(memory_->get_free_blocks() + memory_->get_reserved_blocks()));
  if (need_alloc_blk > 0) {
    tasks.alloc_blks = need_alloc_blk;
  }

  // reserved blocks： decode 使用系统显存也不够时，释放reserved
  int need_free_reserved = sum_decode_blks - int(memory_->get_free_blocks() + system_blocks_);
  // if (!offload2cpu_kvcache_.empty()) {
  //   need_free_reserved = INT32_MAX;
  // }
  //
  if (need_free_reserved > 0) {
    tasks.free_reserved = need_free_reserved;
  }

  return;
}

StepOutput KVCacheSchedule::step() {
  {
    ipipe::TimeGuard guard("KVCacheSchedule: step memory_->wait()");
    memory_->wait();
  }

  ipipe::TimeGuard guard("KVCacheSchedule: step");

  system_blocks_ = memory_->get_system_blocks();
  for (const auto& item : valid_prefill_reqs_) {
    auto iter = prefill_kvcache_.find(item);
    if (iter != prefill_kvcache_.end()) {
      iter->second.seq_len_with_output += 1;
      iter->second.iter_index += 1;
      decode_kvcache_.insert(*iter);
      prefill_kvcache_.erase(iter);
    }
  }
  for (const auto& item : valid_decode_reqs_) {
    auto iter = decode_kvcache_.find(item);
    if (iter != decode_kvcache_.end()) {
      iter->second.seq_len_with_output += 1;
      iter->second.iter_index += 1;
    }
  }

  valid_prefill_reqs_.clear();
  valid_decode_reqs_.clear();

  KVCacheMemory::CachedMemoryTasks tasks;
  prepare_cpu_offload(tasks);

  int free_blocks = memory_->get_free_blocks();
  int reserved_blocks = memory_->get_reserved_blocks();
  // if (free_blocks + reserved_blocks < min_blk_) {
  //   SPDLOG_WARN(
  //       "KVCacheSchedule: Alloc at runtime. free_blocks: {}, reserved_blocks: {}, min_blk: {}",
  //       free_blocks, reserved_blocks, min_blk_);
  //   memory_->alloc_blocks_up_to(min_blk_ - reserved_blocks + 1);
  //   free_blocks = memory_->get_free_blocks();
  //   IPIPE_ASSERT(free_blocks + reserved_blocks >= min_blk_);
  // }
  size_t need_blk = 0;
  size_t valid_blk_size = 0;
  if (decode_kvcache_.empty()) {
    assert(prefill_kvcache_.size() > 0);
    valid_prefill_reqs_ = valid_prefill_requests(
        free_blocks + reserved_blocks, config_.max_batch_size, config_.max_concurrent_requests);
  } else {  // 先看看 decode的情况
    for (const auto& item : decode_kvcache_) {
      if (valid_decode_reqs_.size() >= config_.max_batch_size) break;
      need_blk += cal_decode_blocks(item.second.seq_len_with_output + config_.reserve_decode,
                                    seq_per_block_, config_.layer_num);
      if (need_blk > free_blocks) {  // decode 仅仅用free blocks。reserved blocks已经提前free了
        break;
      }
      valid_blk_size = need_blk;
      valid_decode_reqs_.insert(item.first);
    }
    // decode 均可执行
    if (valid_decode_reqs_.size() == decode_kvcache_.size() &&
        (config_.max_batch_size > valid_decode_reqs_.size()) &&
        config_.max_concurrent_requests > valid_decode_reqs_.size()) {
      if (offload2cpu_kvcache_.empty()) {
        auto tmp =
            valid_prefill_requests(free_blocks + reserved_blocks - need_blk,
                                   config_.max_batch_size - valid_decode_reqs_.size(),
                                   config_.max_concurrent_requests - valid_decode_reqs_.size());
        valid_prefill_reqs_.insert(tmp.begin(), tmp.end());
      }
    } else {
      if (valid_decode_reqs_.empty()) {
        // almost impossible. holy shit, no reserve now, try cpu offload
        // 由于默认decode向前看了一步，所以当前步一定是有空间的
        valid_decode_reqs_.insert(decode_kvcache_.begin()->first);

        // block至少比容纳一个完整request所需块多1个
        valid_blk_size = cal_decode_blocks(
            decode_kvcache_.begin()->second.seq_len_with_output + config_.reserve_decode,
            seq_per_block_, config_.layer_num);
      }
    }
  }

  memory_->alloc(valid_prefill_reqs_, prefill_kvcache_, valid_decode_reqs_, decode_kvcache_);

  prepare_next_memory(tasks);
  memory_->step(tasks);

  auto re = valid_decode_reqs_;
  re.insert(valid_prefill_reqs_.begin(), valid_prefill_reqs_.end());

  for (const auto& item : re) {
    activate(item);
  }
  return re;
}

void KVCacheSchedule::activate(const std::string& request_id) {
  // layer_num[key_input, value_input, key_output, value_output]
  const MemOutput& k = memory_->query_key(request_id);
  const MemOutput& v = memory_->query_value(request_id);

  std::vector<std::vector<torch::Tensor>> re;

  static auto options = get_tensor_options(config_.elemsize);
  auto iter = prefill_kvcache_.find(request_id);
  if (iter != prefill_kvcache_.end()) {
    for (size_t i = 0; i < config_.layer_num; i++) {
      re.push_back({
          torch::from_blob(k[i],
                           {1, iter->second.seq_len_with_output, config_.num_heads,
                            config_.hidden_size / config_.num_heads},
                           options),
          torch::from_blob(v[i],
                           {1, iter->second.seq_len_with_output, config_.num_heads,
                            config_.hidden_size / config_.num_heads},
                           options),
      });
    }
  } else {
    iter = decode_kvcache_.find(request_id);
    IPIPE_ASSERT(iter != decode_kvcache_.end());
    auto seq_offset =
        (config_.hidden_size * config_.elemsize) * (iter->second.seq_len_with_output - 1);
    for (size_t i = 0; i < config_.layer_num; i++) {
      re.push_back(
          {torch::from_blob(k[i],
                            {1, iter->second.seq_len_with_output - 1, config_.num_heads,
                             config_.hidden_size / config_.num_heads},
                            options),
           torch::from_blob(v[i],
                            {1, iter->second.seq_len_with_output - 1, config_.num_heads,
                             config_.hidden_size / config_.num_heads},
                            options),
           torch::from_blob(k[i] + seq_offset,
                            {1, 1, config_.num_heads, config_.hidden_size / config_.num_heads},
                            options),
           torch::from_blob(v[i] + seq_offset,
                            {1, 1, config_.num_heads, config_.hidden_size / config_.num_heads},
                            options)});
    }
  }
  std::shared_ptr<KVCacheV4> data = std::make_shared<KVCacheV4>(std::move(re));

  auto& storage = ipipe::ThreadSafeKVStorage::getInstance().get(request_id);
  storage.set("kvcachev4", data);
}

// void KVCacheSchedule::cache(const CacheParams& ca) { return memory_->cache(ca); }

}  // namespace kvcache