#include <optional>
#include "helper/net_info.hpp"

namespace torchpipe {
inline bool is_all_positive(const NetIOInfo::Dims64& dims) {
    if (dims.nbDims <= 0) return false;
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

size_t elementSize(NetIOInfo::DataType info) {
    switch (info) {
        case NetIOInfo::DataType::INT4:
        case NetIOInfo::DataType::FP4:
            return 0.5;  // 4 bits = 0.5 bytes
        case NetIOInfo::DataType::INT8:
        case NetIOInfo::DataType::UINT8:
        case NetIOInfo::DataType::BOOL:
        case NetIOInfo::DataType::FP8:
            return 1;  // 8 bits = 1 byte
        case NetIOInfo::DataType::INT32:
        case NetIOInfo::DataType::FP32:
        case NetIOInfo::DataType::BF32:
            return 4;  // 32 bits = 4 bytes
        case NetIOInfo::DataType::INT64:
            return 8;  // 64 bits = 8 bytes
        case NetIOInfo::DataType::FP16:
        case NetIOInfo::DataType::BF16:
            return 2;  // 16 bits = 2 bytes
        case NetIOInfo::DataType::RESERVED_INT:
        case NetIOInfo::DataType::RESERVED_FP:
        case NetIOInfo::DataType::RESERVED_BF:
        case NetIOInfo::DataType::UNKNOWN:
        default:
            return 0;  // Unknown or reserved types
    }
}
}  // namespace torchpipe