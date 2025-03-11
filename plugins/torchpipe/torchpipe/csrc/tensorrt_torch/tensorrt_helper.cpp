
#include <sstream>
#include <vector>
#include <numeric>
#include <string>
#include <fstream>

#include <hami/extension.hpp>
#include "tensorrt_torch/tensorrt_helper.hpp"
#include "NvInferPlugin.h"
#include <c10/cuda/CUDAStream.h>
#include <NvOnnxParser.h>
#include <c10/core/ScalarType.h> // Add this line

namespace {

nvinfer1::ILogger::Severity trt_get_log_level(std::string level) {
    if (level == "info")
        return nvinfer1::ILogger::Severity::kINFO;
    else if (level == "error")
        return nvinfer1::ILogger::Severity::kERROR;
    else if (level == "verbose")
        return nvinfer1::ILogger::Severity::kVERBOSE;
    else
        return nvinfer1::ILogger::Severity::kWARNING;
    throw std::invalid_argument(
        "log_level must be one of info, error, verbose, warning");
}
const static std::unordered_set<std::string> FP16_ENABLE{"fp16", "int8", "best",
                                                         "FP16", "INT8"};
const static std::unordered_set<std::string> INT8_ENABLE{"int8", "best",
                                                         "INT8"};

class NvLogger : public nvinfer1::ILogger {
   public:
    void log(Severity severity,
             nvinfer1::AsciiChar const* msg) noexcept override {
        if ((severity == Severity::kERROR) ||
            (severity == Severity::kINTERNAL_ERROR)) {
            spdlog::error(msg);
        } else if (severity == Severity::kINFO)
            // SPDLOG_DEBUG(msg);
            spdlog::debug(msg);
        else if (severity == Severity::kWARNING)
            // SPDLOG_WARN(msg);
            spdlog::warn(msg);
        else if (severity == Severity::kVERBOSE)
            // SPDLOG_TRACE(msg);
            spdlog::trace(msg);
    }
    // #endif
};

}  // namespace

namespace torchpipe {
nvinfer1::ILogger* get_trt_logger() {
    static NvLogger gLogger_inplace;
    return &gLogger_inplace;
}

// https://github.com/maggiez0138/Swin-Transformer-TensorRT/blob/master/trt/trt_utils.py
void force_layernorn_fp32(nvinfer1::INetworkDefinition* network) {
    const static std::string POW_NAME = "Pow";
    const static std::string ReduceMean_NAME = "ReduceMean";
    const static std::string Add_NAME = "Add";
    const static std::set<std::string> LN_NAME{
        POW_NAME, ReduceMean_NAME, Add_NAME, "Sqrt", "Div", "Mul"};

    for (std::size_t index_l = 0; index_l < network->getNbLayers(); ++index_l) {
        nvinfer1::ILayer* layer = network->getLayer(index_l);

        HAMI_ASSERT(layer);
        std::string start_name = layer->getName();
        std::vector<nvinfer1::ILayer*> target;
        bool find_ln = false;
        bool has_reduce_mean = false;
        if (start_name.find(POW_NAME) != std::string::npos &&
            !network->getLayer(index_l)->precisionIsSet()) {
            // auto ln_name = LN_NAME;
            while (index_l < network->getNbLayers()) {
                target.push_back(network->getLayer(index_l));
                SPDLOG_DEBUG("parse {}", network->getLayer(index_l)->getName());
                if (target.size() >= 12) break;
                ++index_l;
                std::string name = network->getLayer(index_l)->getName();
                if (name.find(ReduceMean_NAME) != std::string::npos) {
                    has_reduce_mean = true;
                } else if (name.find("Mul") != std::string::npos) {
                    target.push_back(network->getLayer(index_l));
                    SPDLOG_DEBUG("post Mul {}",
                                 network->getLayer(index_l + 1)->getName());
                    /*
                    cast -> Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul ->
                    cast*/

                    if (has_reduce_mean) find_ln = true;
                    break;
                }
            }
        }
        if (find_ln) {
            std::set<std::string> parsed;
            SPDLOG_INFO("********** LayerNorm matched ************");
            for (std::size_t index = 0; index < target.size(); ++index) {
                std::string name = target[index]->getName();
                target[index]->setPrecision(nvinfer1::DataType::kFLOAT);
                parsed.insert(name);

                if (index != target.size() - 1) {
                    target[index]->setOutputType(0, nvinfer1::DataType::kFLOAT);
                }
                std::string info =
                    "LayerNorm: The following layers were set to fp32 mode: ";
                for (const auto& item : parsed) {
                    info += item + ' ';
                }
                SPDLOG_INFO(info);
            }
            SPDLOG_INFO("*****************************");
        }
    }
}

bool precision_fpx_count(const std::set<std::string>& target,
                         std::string layer_name,
                         std::set<std::string>& layers_founded) {
    layer_name = hami::str::tolower(layer_name);
    for (const auto& item : target) {
        const std::string lower_item = hami::str::tolower(item);
        if (layer_name.find(lower_item) != std::string::npos) {
            layers_founded.insert(item);
            return true;
        }
    }
    return false;
}

void modify_layers_precision(std::set<std::string> precision_fpx_input,
                             nvinfer1::INetworkDefinition* network,
                             nvinfer1::DataType dataType,
                             bool is_output = false) {
    std::set<std::string> precision_fpx;
    std::transform(
        precision_fpx_input.begin(), precision_fpx_input.end(),
        std::inserter(precision_fpx, precision_fpx.begin()),
        [](const std::string& item) { return hami::str::tolower(item); });

    std::set<std::string> layers_founded;

    for (std::size_t index_l = 0;
         !precision_fpx.empty() && index_l < network->getNbLayers();
         ++index_l) {
        nvinfer1::ILayer* layer = network->getLayer(index_l);

        HAMI_ASSERT(layer);

        if (precision_fpx_count(precision_fpx, std::string(layer->getName()),
                                layers_founded)) {
            std::string mode_name = "fp32";
            switch (dataType) {
                case nvinfer1::DataType::kFLOAT:
                    mode_name = "fp32";
                    break;
                case nvinfer1::DataType::kINT8:
                    mode_name = "int8";
                    break;
                case nvinfer1::DataType::kHALF:
                    mode_name = "fp16";
                    break;
                default:
                    throw std::runtime_error("unsupported data type: " +
                                             std::to_string(int(dataType)));
                    break;
            }
            if (is_output) {
                layer->setOutputType(0, dataType);
                SPDLOG_INFO("{}'s output was set to {} mode", layer->getName(),
                            mode_name);
            } else {
                layer->setPrecision(dataType);
                SPDLOG_INFO("{} was set to {} mode", layer->getName(),
                            mode_name);
            }
        }
    }
    for (const auto& item : layers_founded) {
        precision_fpx.erase(item);
    }

    if (!precision_fpx.empty()) {
        std::string error_msg =
            "The following layers were not found in network: ";
        for (const auto& layers_err : precision_fpx) {
            error_msg += layers_err + ' ';
        }
        error_msg += "-----------------\nExisting layers: ";
        for (std::size_t index_l = 0;
             !precision_fpx.empty() && index_l < network->getNbLayers();
             ++index_l) {
            error_msg +=
                std::string(network->getLayer(index_l)->getName()) + "\n";
        }
        throw std::runtime_error(error_msg);
    }
}

void print_colored_net(
    nvinfer1::INetworkDefinition* network,
    const std::vector<int>& input_reorder,
    const std::vector<std::pair<std::string, nvinfer1::Dims>>&
        net_inputs_ordered_dims) {
    std::stringstream ss;

    // Header
    ss << hami::colored(
              "\n==================== Network Inputs ====================")
       << "\n";

    // Print each input's name and dimensions
    for (std::size_t i = 0; i < net_inputs_ordered_dims.size(); ++i) {
        const auto& item = net_inputs_ordered_dims[i];
        ss << hami::colored("Input " + std::to_string(input_reorder[i]) + ": ")
           << hami::colored(item.first) << " [";
        for (int j = 0; j < item.second.nbDims; ++j) {
            const int inputS = item.second.d[j];
            ss << inputS;
            if (j != item.second.nbDims - 1) ss << " x ";
        }
        ss << "]\n";
    }

    // Footer with instructions
    ss << hami::colored(
              "========================================================")
       << "\n";
    ss << hami::colored("Instructions:") << "\n";
    ss << hami::colored(
              "1. Use the above information to set ranges (through parameters: "
              "max/min) for "
              "profiles.")
       << "\n";
    if (input_reorder.size() > 1) {
        ss << hami::colored(
                  "2. Reset the input order by modifying `input_reorder`.")
           << "\n";
    }

    // Print the final output
    SPDLOG_INFO(ss.str());
}

void print_net(nvinfer1::INetworkDefinition* network,
               const std::vector<int>& input_reorder,
               const std::vector<std::pair<std::string, nvinfer1::Dims>>&
                   net_inputs_ordered_dims) {
    // std::stringstream ss;

    // // Header
    // ss << "==================== Network Inputs ====================\n";

    // // Print each input's name and dimensions
    // for (std::size_t i = 0; i < net_inputs_ordered_dims.size(); ++i) {
    //     const auto& item = net_inputs_ordered_dims[i];
    //     ss << "Input " << input_reorder[i] << ": " << item.first << " [";
    //     for (int j = 0; j < item.second.nbDims; ++j) {
    //         const int inputS = item.second.d[j];
    //         ss << inputS;
    //         if (j != item.second.nbDims - 1) ss << " x ";
    //     }
    //     ss << "]\n";
    // }

    // // Print current input order
    // ss << "\nCurrent Input Order: (";
    // for (std::size_t i = 0; i < input_reorder.size(); ++i) {
    //     ss << input_reorder[i];
    //     if (i != input_reorder.size() - 1) ss << ", ";
    // }
    // ss << ")\n";

    // // Footer with instructions
    // ss << "========================================================\n";
    // ss << "Instructions:\n";
    // ss << "1. Use the above information to set ranges (batch sizes) for "
    //       "profiles.\n";
    // if (input_reorder.size() > 1) {
    //     ss << "2. Reset the input order by modifying `input_reorder`.\n";
    // }

    // // Print the final output (assuming SPDLOG_INFO and colored functions are
    // // defined)
    // SPDLOG_INFO(hami::colored(ss.str()));
}
c10::ScalarType trt2torch_type(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return c10::ScalarType::Float;
        case nvinfer1::DataType::kINT32:
            return c10::ScalarType::Int;
        case nvinfer1::DataType::kINT8:
            return c10::ScalarType::Char; // 或 c10::ScalarType::Byte
        case nvinfer1::DataType::kINT64:
            return c10::ScalarType::Long;
        case nvinfer1::DataType::kBOOL:
            return c10::ScalarType::Bool;
        case nvinfer1::DataType::kHALF:
            return c10::ScalarType::Half;
        default:
            SPDLOG_ERROR(
                "Unsupported data type: only support kFLOAT, kINT32, kINT64, kINT8, "
                "kBOOL, kHALF");
            throw std::runtime_error("Unsupported datatype");
    }
}

// Helper function to apply mean and std normalization
nvinfer1::ITensor* mean_std_helper(nvinfer1::INetworkDefinition* network,
                                   nvinfer1::ITensor* input, const float* mean,
                                   const float* std,
                                   std::set<nvinfer1::ILayer*>& new_layers,
                                   bool set_half) {
    // Validate mean and std values
    for (int i = 0; i < 3; ++i) {
        if (mean && mean[i] <= 1 + 1e-5) {
            throw std::invalid_argument(
                "Mean values must be greater than 1 + 1e-5.");
        }
        if (std && std[i] <= 1 + 1e-5) {
            throw std::invalid_argument(
                "Std values must be greater than 1 + 1e-5.");
        }
    }

    nvinfer1::ITensor* itensor = input;

    // Apply mean subtraction
    if (mean) {
        nvinfer1::Weights Mean{nvinfer1::DataType::kFLOAT, mean, 3};
        nvinfer1::IConstantLayer* m =
            network->addConstant(nvinfer1::Dims4{1, 3, 1, 1}, Mean);
        new_layers.insert(m);

        auto* sub_mean = network->addElementWise(
            *itensor, *m->getOutput(0), nvinfer1::ElementWiseOperation::kSUB);
        new_layers.insert(sub_mean);
        itensor = sub_mean->getOutput(0);
    }

    // Apply std division
    if (std) {
        nvinfer1::Weights Std{nvinfer1::DataType::kFLOAT, std, 3};
        nvinfer1::IConstantLayer* s =
            network->addConstant(nvinfer1::Dims4{1, 3, 1, 1}, Std);
        nvinfer1::IElementWiseLayer* std_mean = network->addElementWise(
            *itensor, *s->getOutput(0), nvinfer1::ElementWiseOperation::kDIV);
        new_layers.insert(s);
        new_layers.insert(std_mean);
        itensor = std_mean->getOutput(0);
    }

    return itensor;
}

// Function to merge mean and std into the network
void merge_mean_std(nvinfer1::INetworkDefinition* network,
                    const std::vector<float>& mean,
                    const std::vector<float>& std) {
    // Validate input tensor
    nvinfer1::ITensor* input = network->getInput(0);
    if (!input) {
        throw std::invalid_argument(
            "Network must have at least one input tensor.");
    }

    // Validate mean and std vectors
    if (mean.size() != 3 && !mean.empty()) {
        throw std::invalid_argument(
            "Mean vector must have exactly 3 elements or be empty.");
    }
    if (std.size() != 3 && !std.empty()) {
        throw std::invalid_argument(
            "Std vector must have exactly 3 elements or be empty.");
    }

    // Check if mean or std is provided
    if (!mean.empty() || !std.empty()) {
        SPDLOG_WARN("Start merging mean/std into the network.");

        // Create a set to track new layers
        std::set<nvinfer1::ILayer*> new_layers;

        // Merge mean and std into the network
        nvinfer1::ITensor* pre_input = mean_std_helper(
            network, input, mean.empty() ? nullptr : mean.data(),
            std.empty() ? nullptr : std.data(), new_layers, false);

        // Update network layers to use the preprocessed input
        for (int i = 0; i < network->getNbLayers(); ++i) {
            auto* layer = network->getLayer(i);
            if (new_layers.find(layer) != new_layers.end()) continue;
            for (int j = 0; j < layer->getNbInputs(); ++j) {
                if (layer->getInput(j) == input) {
                    layer->setInput(j,
                                    *pre_input);  // Replace the original input
                    break;
                }
            }
        }
    }
}

bool initTrtPlugins() {
    static bool didInitPlugins = initLibNvInferPlugins(get_trt_logger(), "");
    assert(didInitPlugins);
    return didInitPlugins;
}

nvinfer1::Dims infer_shape(std::vector<int> config_shape,
                           const nvinfer1::Dims& net_input) {
    nvinfer1::Dims out = net_input;
    HAMI_ASSERT(config_shape.size() == net_input.nbDims);
    for (std::size_t i = 0; i < net_input.nbDims; ++i) {
        if (config_shape[i] == -1) {
            if (-1 == net_input.d[i]) {
                throw std::invalid_argument(
                    "Both the network input and the configuration input are "
                    "set to -1, "
                    "making it impossible to automatically infer the shape of "
                    "this dimension.");
            } else {
                config_shape[i] = net_input.d[i];
            }
        } else if (net_input.d[i] != -1 && net_input.d[i] != config_shape[i]) {
            const std::string error_msg =
                "shape from network and shape from configuration not match: "
                "net_input[" +
                std::to_string(i) + "]= " + std::to_string(net_input.d[i]) +
                " input= " + std::to_string(config_shape[i]);
            SPDLOG_ERROR(error_msg);
            throw std::invalid_argument(error_msg);
        }
        out.d[i] = config_shape[i];
    }
    return out;
}

void update_min_max_setting(
    std::vector<std::vector<std::vector<int>>>& mins,
    std::vector<std::vector<std::vector<int>>>& maxs,
    const std::vector<std::pair<std::string, nvinfer1::Dims>>&
        net_inputs_ordered_dims) {
    // Get the number of network inputs
    size_t net_inputs = net_inputs_ordered_dims.size();

    // Determine the number of profiles based on max of mins and maxs sizes
    size_t profile_num = std::max(mins.size(), maxs.size());
    if (profile_num == 0) {
        // Create a single profile by default
        mins.resize(1);
        maxs.resize(1);
        profile_num = 1;
    } else {
        // Ensure both mins and maxs have the same number of profiles
        mins.resize(profile_num);
        maxs.resize(profile_num);
    }

    // Process each profile
    for (size_t p = 0; p < profile_num; ++p) {
        auto& min_profile = mins[p];
        auto& max_profile = maxs[p];

        // Ensure each profile has entries for all network inputs
        if (min_profile.size() < net_inputs) {
            min_profile.resize(net_inputs);
        }
        if (max_profile.size() < net_inputs) {
            max_profile.resize(net_inputs);
        }

        // Process each input
        for (size_t i = 0; i < net_inputs; ++i) {
            const auto& input_dims = net_inputs_ordered_dims[i].second;
            HAMI_ASSERT(input_dims.nbDims > 0,
                        "Input " + std::to_string(i) + " has no dimensions");

            // Check if this input's dimensions are configured in the current
            // profile
            if (min_profile[i].size() < input_dims.nbDims) {
                min_profile[i].resize(input_dims.nbDims, -1);
            }
            if (max_profile[i].size() < input_dims.nbDims) {
                max_profile[i].resize(input_dims.nbDims, -1);
            }

            // Process each dimension in the input
            for (int d = 0; d < input_dims.nbDims; ++d) {
                int current_net_dim = input_dims.d[d];

                if (current_net_dim == -1) {
                    // Handle dynamic dimensions
                    if (min_profile[i][d] == -1) {
                        min_profile[i][d] = 1;
                    }
                    if (max_profile[i][d] == -1) {
                        max_profile[i][d] = 1;
                    }
                } else {
                    // Use network's dimension as default if not configured
                    if (min_profile[i][d] == -1) {
                        min_profile[i][d] = current_net_dim;
                    }
                    if (max_profile[i][d] == -1) {
                        max_profile[i][d] = current_net_dim;
                    }
                    HAMI_ASSERT(max_profile[i][d] == current_net_dim &&
                                    min_profile[i][d] == current_net_dim,
                                "For input " + std::to_string(i) +
                                    ", dimension " + std::to_string(d) +
                                    ": config value (" +
                                    std::to_string(max_profile[i][d]) +
                                    ") must match network dimension (" +
                                    std::to_string(current_net_dim) + ")");
                }

                // Ensure max >= min
                if (max_profile[i][d] < min_profile[i][d]) {
                    max_profile[i][d] = min_profile[i][d];
                }
            }
        }
    }

    // Validate the configurations
    for (size_t p = 0; p < profile_num; ++p) {
        // Check for matching input and dimension sizes in each profile
        HAMI_ASSERT(
            mins[p].size() == net_inputs,
            "Profile " + std::to_string(p) + " has mismatched input size");
        HAMI_ASSERT(
            maxs[p].size() == net_inputs,
            "Profile " + std::to_string(p) + " has mismatched input size");

        for (size_t i = 0; i < net_inputs; ++i) {
            const auto& input_dims = net_inputs_ordered_dims[i].second;
            HAMI_ASSERT(input_dims.nbDims == mins[p][i].size(),
                        "Input " + std::to_string(i) + " in profile " +
                            std::to_string(p) +
                            " has mismatched dimension size");
            HAMI_ASSERT(input_dims.nbDims == maxs[p][i].size(),
                        "Input " + std::to_string(i) + " in profile " +
                            std::to_string(p) +
                            " has mismatched dimension size");

            // Check max >= min for each dimension
            for (int d = 0; d < input_dims.nbDims; ++d) {
                HAMI_ASSERT(maxs[p][i][d] >= mins[p][i][d],
                            "For profile " + std::to_string(p) + ", input " +
                                std::to_string(i) + ", dimension " +
                                std::to_string(d) + ": max (" +
                                std::to_string(maxs[p][i][d]) +
                                ") must be >= min (" +
                                std::to_string(mins[p][i][d]) + ")");
            }
        }
    }
}

std::unique_ptr<nvinfer1::IHostMemory> onnx2trt(OnnxParams& params) {
    HAMI_ASSERT(initTrtPlugins());
    const auto explicitBatch =
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::IBuilder> builder{
        nvinfer1::createInferBuilder(*get_trt_logger())};
    std::unique_ptr<nvinfer1::INetworkDefinition> network{
        builder->createNetworkV2(explicitBatch)};
    std::unique_ptr<nvonnxparser::IParser> parser{
        nvonnxparser::createParser(*network, *get_trt_logger())};
    std::unique_ptr<nvinfer1::IBuilderConfig> config{
        builder->createBuilderConfig()};
    size_t max_threads = std::thread::hardware_concurrency();
    if (max_threads <= 5) {
        max_threads = 1;
    } else {
        max_threads = max_threads / 2 - 1;
    }
    if (builder->setMaxThreads(max_threads))
        SPDLOG_INFO("tensorrt builder: max_threads={}", max_threads);

    // todo timecache
    auto b_parsed = parser->parseFromFile(
        params.model.c_str(),
        static_cast<int>(trt_get_log_level(params.log_level)));
    HAMI_ASSERT(b_parsed);
    // todo max workspace size for setMemoryPoolLimit

    bool use_only_fp32 = true;
    // quantize
    if (FP16_ENABLE.count(params.precision) != 0 &&
        builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        use_only_fp32 = false;
    }
    if (INT8_ENABLE.count(params.precision) != 0 &&
        builder->platformHasFastInt8()) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        use_only_fp32 = false;
    }
    // modify_layers_precision
    modify_layers_precision(params.precision_fp32, network.get(),
                            nvinfer1::DataType::kFLOAT);
    modify_layers_precision(params.precision_fp16, network.get(),
                            nvinfer1::DataType::kHALF);

    if ((!use_only_fp32) && params.force_layer_norm_pattern_fp32)
        force_layernorn_fp32(network.get());

    // todo reorder
    std::vector<int> input_reorder(network->getNbInputs());
    std::iota(input_reorder.begin(), input_reorder.end(), 0);

    // get dims
    std::vector<std::pair<std::string, nvinfer1::Dims>> net_inputs_ordered_dims;
    for (int i = 0; i < input_reorder.size(); ++i) {
        net_inputs_ordered_dims.push_back(
            {network->getInput(input_reorder[i])->getName(),
             network->getInput(input_reorder[i])->getDimensions()});
    }

    print_colored_net(network.get(), input_reorder, net_inputs_ordered_dims);
    merge_mean_std(network.get(), params.mean, params.std);

    // profile
    update_min_max_setting(params.mins, params.maxs, net_inputs_ordered_dims);
    const auto profile_num = params.mins.size();
    nvinfer1::IOptimizationProfile* first_profile = nullptr;
    for (size_t index_p = 0; index_p < profile_num; ++index_p) {
        auto profile = builder->createOptimizationProfile();
        if (!first_profile) first_profile = profile;

        // mins: multiple profiles x multiple inputs x multiDims
        // if (params.mins[index_p].size() < network->getNbInputs()) {
        //     HAMI_ASSERT(!params.mins[index_p].empty());
        //     params.mins[index_p].resize(network->getNbInputs(),
        //                                 params.mins[index_p].back());
        // }
        // if (params.maxs[index_p].size() < network->getNbInputs()) {
        //     HAMI_ASSERT(!params.maxs[index_p].empty());
        //     params.maxs[index_p].resize(network->getNbInputs(),
        //                                 params.maxs[index_p].back());
        // }
        HAMI_ASSERT(params.mins[index_p].size() == network->getNbInputs());
        HAMI_ASSERT(params.maxs[index_p].size() == network->getNbInputs());

        for (int i = 0; i < input_reorder.size(); ++i) {
            if (network->getInput(input_reorder[i])->isShapeTensor()) {
                auto& min_dim = params.mins[index_p][i];
                auto& max_dim = params.maxs[index_p][i];

                // todo check input dims match
                HAMI_ASSERT(profile->setShapeValues(
                    network->getInput(input_reorder[i])->getName(),
                    nvinfer1::OptProfileSelector::kMIN, min_dim.data(),
                    min_dim.size()));
                HAMI_ASSERT(profile->setShapeValues(
                    network->getInput(input_reorder[i])->getName(),
                    nvinfer1::OptProfileSelector::kMAX, max_dim.data(),
                    max_dim.size()));
                HAMI_ASSERT(profile->setShapeValues(
                    network->getInput(input_reorder[i])->getName(),
                    nvinfer1::OptProfileSelector::kOPT, max_dim.data(),
                    max_dim.size()));

                continue;
            }
            auto net_shape =
                network->getInput(input_reorder[i])->getDimensions();
            auto min_dim = infer_shape(params.mins[index_p][i], net_shape);
            auto max_dim = infer_shape(params.maxs[index_p][i], net_shape);
            profile->setDimensions(
                network->getInput(input_reorder[i])->getName(),
                nvinfer1::OptProfileSelector::kMIN, min_dim);
            profile->setDimensions(
                network->getInput(input_reorder[i])->getName(),
                nvinfer1::OptProfileSelector::kOPT, max_dim);
            profile->setDimensions(
                network->getInput(input_reorder[i])->getName(),
                nvinfer1::OptProfileSelector::kMAX, max_dim);
            if (!(max_dim.nbDims > 0 && min_dim.nbDims > 0)) {
                SPDLOG_ERROR(
                    "max_dim.nbDims = {} net_shape.nbDims={} check failed: "
                    "max_dim.nbDims > 0 && "
                    "min_dim.nbDims > 0",
                    max_dim.nbDims, min_dim.nbDims, net_shape.nbDims);
                throw std::invalid_argument(
                    "max_dim.nbDims > 0 && min_dim.nbDims > 0");
            }
            // todo dynamic batch size check
        }
        config->addOptimizationProfile(profile);
    }

#if (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 1) || \
    (NV_TENSORRT_MAJOR >= 11)
    // config->setFlag(nvinfer1::BuilderFlag::kWEIGHT_STREAMING);
#endif

    // build engine
    SPDLOG_INFO(
        "Building engine with {} profiles and precision={}... (this may take "
        "some time)",
        profile_num, params.precision);

    auto time_now = hami::now();
    std::unique_ptr<nvinfer1::IHostMemory> engine_plan(
        builder->buildSerializedNetwork(*network, *config));
    HAMI_ASSERT(engine_plan->size() > 0);
    auto time_pass = hami::time_passed(time_now);
    SPDLOG_INFO("Engine building completed in {:.2f} seconds",
                time_pass / 1000.0);
    if (params.model_cache.size() > 0) {
        SPDLOG_INFO("Saving engine to {}", params.model_cache);
        std::ofstream file(params.model_cache, std::ios::binary);
        file.write(static_cast<char*>(engine_plan->data()),
                   engine_plan->size());
        // engine_plan.release();
    }
    return engine_plan;
}

OnnxParams config2onnxparams(
    const std::unordered_map<std::string, std::string>& config) {
    OnnxParams params;

    hami::str::try_update(config, "model", params.model);
    hami::str::try_update(config, "model::cache", params.model_cache);

    hami::str::try_update(config, "max_workspace_size",
                          params.max_workspace_size);
    HAMI_ASSERT(
        params.max_workspace_size >= 1 &&
            params.max_workspace_size < 1'000'000'000,
        "max_workspace_size must be in MB and between 1 and 1,000,000,000");
    params.max_workspace_size = 1024 * 1024 * params.max_workspace_size;

    hami::str::try_update(config, "model::timingcache", params.timingcache);

    hami::str::try_update(config, "log_level", params.log_level);

    // 处理精度相关参数
    hami::str::try_update(config, "precision", params.precision);
    if (params.precision.empty()) {
        // params.precision = "fp16";
        // auto sm = get_sm();
        // if (sm <= "6.1")
        //     params.precision = "fp32";
        // else
        //     params.precision = "fp16";
        // SPDLOG_WARN(
        //     "'precision' not set. You can set it to one of "
        //     "[fp16|fp32|int8|best]. Default to fp16 if "
        //     "platformHasFastFp16 and SM>6.1 else fp32.\n");
    }

    // 处理精度层设置
    if (config.find("precision::fp32") != config.end()) {
        auto precision_fp32 =
            hami::str::str_split(config.at("precision::fp32"), ',');
        params.precision_fp32 =
            std::set<std::string>(precision_fp32.begin(), precision_fp32.end());
        SPDLOG_INFO("these layers keep fp32: {}", config.at("precision::fp32"));
    }

    if (config.find("precision::fp16") != config.end()) {
        auto precision_fp16 =
            hami::str::str_split(config.at("precision::fp16"), ',');
        params.precision_fp16 =
            std::set<std::string>(precision_fp16.begin(), precision_fp16.end());
        SPDLOG_INFO("these layers keep fp16: {}", config.at("precision::fp16"));
    }

    // 处理其他参数
    hami::str::try_update(config, "force_layer_norm_pattern_fp32",
                          params.force_layer_norm_pattern_fp32);

    // 处理 mean 和 std
    if (config.find("mean") != config.end()) {
        params.mean = hami::str::str_split<float>(config.at("mean"));
    }
    if (config.find("std") != config.end()) {
        params.std = hami::str::str_split<float>(config.at("std"));
    }

    // 处理 min 和 max shapes
    if (config.find("min") != config.end() && !config.at("min").empty()) {
        auto min_shapes = hami::str::str_split(config.at("min"), ';');
        params.maxs =
            hami::str::str_split<int>(config.at("min"), 'x', ',', ';');
    }

    if (config.find("max") != config.end() && !config.at("max").empty()) {
        params.maxs =
            hami::str::str_split<int>(config.at("max"), 'x', ',', ';');
    }

    return params;
}
static NetIOInfo::DataType convert_type(const nvinfer1::DataType& data_type) {
    // NetIOInfo::DataType target_data_type;
    switch (data_type) {
        case nvinfer1::DataType::kFLOAT:
            return NetIOInfo::DataType::FP32;

        case nvinfer1::DataType::kINT32:
            return NetIOInfo::DataType::INT32;

        case nvinfer1::DataType::kINT8:
            return NetIOInfo::DataType::INT8;

        case nvinfer1::DataType::kHALF:
            return NetIOInfo::DataType::FP16;
        default:
            break;
    }
    throw std::runtime_error("unsupported data type: " +
                             std::to_string(int(data_type)));
};

NetIOInfos get_context_shape(nvinfer1::IExecutionContext* context,
                             size_t profile_index) {
    static_assert(sizeof(nvinfer1::Dims) == sizeof(NetIOInfo::Dims64));
    // NetIOInfos io_info;
    const nvinfer1::ICudaEngine& engine = context->getEngine();
    const auto num_inputsOutputs = engine.getNbIOTensors();

    std::vector<NetIOInfo> io_infos(num_inputsOutputs);

    size_t num_input = 0;

    for (int j = 0; j < num_inputsOutputs; j++) {
        const auto name = engine.getIOTensorName(j);
        const auto tensorType = engine.getTensorIOMode(name);
        const auto dataType = convert_type(engine.getTensorDataType(name));

        io_infos[j].name = name;
        io_infos[j].type = dataType;
        if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
            nvinfer1::Dims min_dims = engine.getProfileShape(
                name, profile_index, nvinfer1::OptProfileSelector::kMIN);
            HAMI_ASSERT(context->setInputShape(name, min_dims));
            memcpy(&io_infos[j].min, &min_dims, sizeof(nvinfer1::Dims));
            num_input++;
        }
    }
    for (int j = 0; j < num_inputsOutputs; j++) {
        const auto name = engine.getIOTensorName(j);
        const auto tensorType = engine.getTensorIOMode(name);
        if (tensorType == nvinfer1::TensorIOMode::kOUTPUT) {
            nvinfer1::Dims dims = context->getTensorShape(name);
            memcpy(&io_infos[j].min, &dims, sizeof(nvinfer1::Dims));
        }
    }
    for (int j = 0; j < num_inputsOutputs; j++) {
        const auto name = engine.getIOTensorName(j);
        const auto tensorType = engine.getTensorIOMode(name);
        if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
            nvinfer1::Dims max_dims = engine.getProfileShape(
                name, profile_index, nvinfer1::OptProfileSelector::kMAX);
            HAMI_ASSERT(context->setInputShape(name, max_dims));
            memcpy(&io_infos[j].max, &max_dims, sizeof(nvinfer1::Dims));
        }
    }
    for (int j = 0; j < num_inputsOutputs; j++) {
        const auto name = engine.getIOTensorName(j);
        const auto tensorType = engine.getTensorIOMode(name);
        if (tensorType == nvinfer1::TensorIOMode::kOUTPUT) {
            nvinfer1::Dims dims = context->getTensorShape(name);
            memcpy(&io_infos[j].max, &dims, sizeof(nvinfer1::Dims));
        }
    }
    return {{io_infos.begin(), io_infos.begin() + num_input},
            {io_infos.begin() + num_input, io_infos.end()}};
};

std::unique_ptr<nvinfer1::IExecutionContext> create_context(
    nvinfer1::ICudaEngine* engine, size_t instance_index) {
    const auto num_profiles = engine->getNbOptimizationProfiles();
    HAMI_ASSERT(instance_index < num_profiles);

#if TRT_USER_MANAGED_MEM
    // USE_OUT_MEM
    std::unique_ptr<nvinfer1::IExecutionContext> context =
        std::unique_ptr<nvinfer1::IExecutionContext>(
            engine->createExecutionContext(
                nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));

#else
    std::unique_ptr<nvinfer1::IExecutionContext> context =
        unique_ptr<nvinfer1::IExecutionContext>(
            engine->createExecutionContext());
#endif

    context->setOptimizationProfileAsync(instance_index,
                                         c10::cuda::getCurrentCUDAStream());

    return context;
}

}  // namespace torchpipe