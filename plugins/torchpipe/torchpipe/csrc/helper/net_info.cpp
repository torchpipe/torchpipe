#include "helper/net_info.hpp"

namespace torchpipe {
inline bool is_all_positive(const NetIOInfo::Dims64& dims) {
    for (size_t index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] <= 0) return false;
    }
    return true;
}

bool is_all_positive(NetIOInfos& info) {
    for (const auto& item : info.first) {
        if (!is_all_positive(item.min)) return false;
    }
    for (const auto& item : info.second) {
        if (!is_all_positive(item.min)) return false;
    }
    for (const auto& item : info.first) {
        if (!is_all_positive(item.max)) return false;
    }
    for (const auto& item : info.second) {
        if (!is_all_positive(item.max)) return false;
    }

    return true;
}
}  // namespace torchpipe