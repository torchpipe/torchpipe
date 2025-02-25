#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace torchpipe {
struct NetIOInfo {
    enum class DataType {
        RESERVED_INT = 0,
        INT4,
        INT8,
        UINT8,
        INT32,
        INT64,
        BOOL,

        RESERVED_FP = 32,
        FP4,
        FP8,
        FP32,
        FP16,

        RESERVED_BF = 48,
        BF16,
        BF32,

        UNKNOWN = 255
    };

    struct Dims64 {
        // The maximum rank (number of dimensions) supported for a tensor.
       public:
        //! The maximum rank (number of dimensions) supported for a tensor.
        static constexpr int32_t MAX_DIMS{8};

        //! The rank (number of dimensions).
        int32_t nbDims;

        //! The extent of each dimension.
        int64_t d[MAX_DIMS];
    };

    Dims64 min;
    Dims64 max;
    DataType type{DataType::FP32};
    std::optional<std::string> name;
};

using NetIOInfos = std::pair<std::vector<NetIOInfo>, std::vector<NetIOInfo>>;

bool is_all_positive(NetIOInfos& info);
}  // namespace torchpipe