#pragma once
#include <cstddef>

static inline int cal_prefill_blocks(size_t seq_len_with_out, size_t seq_per_block,
                                     size_t layer_num) {
  size_t blk_need = seq_len_with_out / seq_per_block;
  if (seq_len_with_out % seq_per_block != 0) {
    blk_need += 1;
  }
  return blk_need * 2 * layer_num;
}
static inline int cal_decode_blocks(size_t seq_len_with_out, size_t seq_per_block,
                                    size_t layer_num) {
  return (seq_len_with_out % seq_per_block == 1) ? 2 * layer_num : 0;
}