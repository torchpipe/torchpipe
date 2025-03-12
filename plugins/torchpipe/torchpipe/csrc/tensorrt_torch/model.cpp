
#include <filesystem>
#include <fstream>
#include "hami/helper/macro.h"
#include "tensorrt_torch/model.hpp"
#include "tensorrt_torch/allocator.hpp"
#include "tensorrt_torch/tensorrt_helper.hpp"

namespace {

#if NV_TENSORRT_MAJOR > 10 || \
    (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 7)
#define USE_TENSORRT_STREAMER_V2
#endif

#ifdef USE_TENSORRT_STREAMER_V2
class LocalFileStreamReader : public nvinfer1::IStreamReaderV2
#else
class LocalFileStreamReader : public nvinfer1::IStreamReader
#endif
{
   private:
    std::ifstream file_stream;
    std::string file_path;

   public:
    explicit LocalFileStreamReader(const std::string& path) : file_path(path) {
        file_stream.open(file_path, std::ios::binary | std::ios::in);
        if (!file_stream.is_open()) {
            throw std::runtime_error("Failed to open file: " + file_path);
        }
    }

    std::vector<char> read() {
        if (!file_stream.is_open() || !file_stream.good()) {
            throw std::runtime_error("Cannot read from file: " + file_path);
        }

        // 保存当前位置
        auto current_pos = file_stream.tellg();

        // 移动到文件末尾以获取文件大小
        file_stream.seekg(0, std::ios::end);
        auto file_size = file_stream.tellg();

        // 回到文件开始位置
        file_stream.seekg(0, std::ios::beg);

        std::vector<char> buffer(file_size);

        if (!file_stream.read(buffer.data(), file_size)) {
            throw std::runtime_error("Failed to read file: " + file_path);
        }

        file_stream.seekg(current_pos);

        return buffer;
    }

    ~LocalFileStreamReader() override {
        if (file_stream.is_open()) {
            file_stream.close();
        }
    }

#ifdef USE_TENSORRT_STREAMER_V2
    bool seek(int64_t offset,
              nvinfer1::SeekPosition where) noexcept override final {
        switch (where) {
            case (nvinfer1::SeekPosition::kSET):
                file_stream.seekg(offset, std::ios_base::beg);
                break;
            case (nvinfer1::SeekPosition::kCUR):
                file_stream.seekg(offset, std::ios_base::cur);
                break;
            case (nvinfer1::SeekPosition::kEND):
                file_stream.seekg(offset, std::ios_base::end);
                break;
        }
        return file_stream.good();
    }

    int64_t read(void* destination, int64_t nbBytes,
                 cudaStream_t stream) noexcept override final {
        if (!file_stream.good()) {
            return -1;
        }

        cudaPointerAttributes attributes;
        if (cudaPointerGetAttributes(&attributes, destination) != cudaSuccess)
            return -1;

        // from CUDA 11 onward, host pointers are return
        // cudaMemoryTypeUnregistered
        if (attributes.type == cudaMemoryTypeHost ||
            attributes.type == cudaMemoryTypeUnregistered) {
            file_stream.read(static_cast<char*>(destination), nbBytes);
            return file_stream.gcount();
        } else if (attributes.type == cudaMemoryTypeDevice) {
            // Set up a temp buffer to read into if reading into device memory.
            std::unique_ptr<char[]> tmpBuf{new char[nbBytes]};
            file_stream.read(tmpBuf.get(), nbBytes);
            // cudaMemcpyAsync into device storage.
            if (cudaMemcpyAsync(destination, tmpBuf.get(), nbBytes,
                                cudaMemcpyHostToDevice, stream) != cudaSuccess)
                return -1;
            // ok to free tmpBuf
            // cudaMemcpyAsync will return once the pageable buffer has been.
            return file_stream.gcount();
        }
        return -1;
    }

    void reset() {
        HAMI_ASSERT(file_stream.good());
        file_stream.seekg(0);
    }
#else
    int64_t read(void* destination, int64_t nbBytes) override {
        if (!file_stream.good()) {
            return -1;
        }
        file_stream.read(static_cast<char*>(destination), nbBytes);
        return file_stream.gcount();
    }
#endif
};

// }

}  // namespace
namespace torchpipe {

void LoadTensorrtEngine::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const hami::dict& dict_config) {
    HAMI_ASSERT(dict_config);
    if (dict_config->find(TASK_ENGINE_KEY) != dict_config->end()) {
        HAMI_ASSERT(dict_config->at(TASK_ENGINE_KEY).type() ==
                    typeid(nvinfer1::ICudaEngine*));
        SPDLOG_INFO(
            "LoadTensorrtEngine: aready loaded engine in dict_config, skip "
            "loading.");
        return;
    }
    // handle instance index
    int instance_num{1};
    hami::str::try_update(config, "instance_num", instance_num);
    HAMI_ASSERT(instance_num >= 1);

    // initialize converter, get std::shared_ptr<ICudaEngine>
    HAMI_ASSERT(config.find("model") != config.end(),
                "`model` is not found in config");

    HAMI_ASSERT(hami::filesystem::exists(config.at("model")));
    LocalFileStreamReader reader(config.at("model"));
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(*get_trt_logger()));
    allocator_ = std::make_unique<TorchAsyncAllocator>();
    runtime_->setGpuAllocator(allocator_.get());
    auto data = reader.read();  // core dump if directly use reader...
    auto* engine_ptr =
        runtime_->deserializeCudaEngine(data.data(), data.size());
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(engine_ptr);
    HAMI_ASSERT(engine_->getNbOptimizationProfiles() == instance_num);

    (*dict_config)[TASK_ENGINE_KEY] = engine_ptr;
    // engine_ = nullptr;
}

void Onnx2Tensorrt::impl_init(
    const std::unordered_map<std::string, std::string>& config,
    const hami::dict& dict_config) {
    HAMI_ASSERT(dict_config);
    if (dict_config->find(TASK_ENGINE_KEY) != dict_config->end()) {
        HAMI_ASSERT(dict_config->at(TASK_ENGINE_KEY).type() ==
                    typeid(nvinfer1::ICudaEngine*));
        SPDLOG_INFO(
            "Onnx2Tensorrt: aready loaded engine in dict_config, skip "
            "loading.");
        return;
    }
    // handle instance index
    // int instance_num{1};
    // hami::str::try_update(config, "instance_num", instance_num);
    // // HAMI_ASSERT(instance_num >= 1 && instance_index_ == 0);
    // HAMI_ASSERT(instance_num >= 1);

    // set runtime && allocator
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(*get_trt_logger()));
    allocator_ = std::make_unique<TorchAsyncAllocator>();
    runtime_->setGpuAllocator(allocator_.get());

    // initialize converter, get std::shared_ptr<ICudaEngine>
    HAMI_ASSERT(config.find("model") != config.end() ||
                    config.find("model::cache") != config.end(),
                "Neither `model` nor `model::cache` is found in config");

    OnnxParams params = config2onnxparams(config);

    bool model_cache_exist = hami::filesystem::exists(params.model_cache);
    // auto mem = !model_cache_exist ? onnx2trt(params) : nullptr;

    if (!model_cache_exist) {
        HAMI_ASSERT(
            hami::filesystem::exists(params.model),
            "file of `model(and model::cache)` not found: " + params.model);
        auto mem = onnx2trt(params);
        HAMI_ASSERT(mem);
        if (!params.model_cache.empty()) {
            std::ofstream ff(params.model_cache, std::ios::binary);
            ff.write((char*)mem->data(), mem->size());

            SPDLOG_INFO(
                "engine saved to {}. Now start to deserializeCudaEngine",
                params.model_cache);
        }

        auto* engine_ptr =
            runtime_->deserializeCudaEngine(mem->data(), mem->size());
        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(engine_ptr);
        HAMI_ASSERT(engine_ && engine_->getNbOptimizationProfiles() ==
                                   params.instance_num);

    } else {
        SPDLOG_INFO(
            "load engine from cache {}, delete it if "
            "you want to rebuild engine.",
            params.model_cache);
        LocalFileStreamReader reader(params.model_cache);
        auto data = reader.read();  // core dump without this...
        auto* engine_ptr =
            runtime_->deserializeCudaEngine(data.data(), data.size());
        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(engine_ptr);
        HAMI_ASSERT(engine_ && engine_->getNbOptimizationProfiles() ==
                                   params.instance_num);
    }

    (*dict_config)[TASK_ENGINE_KEY] = engine_.get();
}

void ModelLoadder::post_init(
    const std::unordered_map<std::string, std::string>& config,
    const hami::dict& dict_config) {
    HAMI_ASSERT(dict_config);
    auto iter = dict_config->find(TASK_ENGINE_KEY);
    if (iter != dict_config->end()) return;

    std::string model_type;
    if (config.find("model_type") != config.end()) {
        model_type = config.at("model_type");
    }

    std::unordered_map<std::string, std::string> suffix_config;
    Backend* backend{nullptr};
    for (size_t i = 0; i < base_config_.size(); ++i) {
        const auto& filter = base_config_[i].at("filter");

        if ((model_type == filter) || ("." + model_type == filter) ||
            (config.find("model") != config.end() &&
             hami::str::endswith(config.at("model"), filter))) {
            backend = base_dependencies_[i].get();
            lazy_init_func_[i]();
            break;
        }
    }
    HAMI_ASSERT(
        backend,
        "ModelLoader: You must set one of the parameters `model_type` or "
        "`model`. We will select the appropriate model loader based on the "
        "precondition from "
        "ModelLoader[(model_suffix_a)A, (model_suffix_b)B].");
    max_ = backend->max();
    min_ = backend->min();

    iter = dict_config->find(TASK_ENGINE_KEY);
    HAMI_ASSERT(iter != dict_config->end());
}

HAMI_REGISTER(hami::Backend, ModelLoadder);
HAMI_REGISTER(hami::Backend, Onnx2Tensorrt);
HAMI_REGISTER(hami::Backend, LoadTensorrtEngine);
}  // namespace torchpipe