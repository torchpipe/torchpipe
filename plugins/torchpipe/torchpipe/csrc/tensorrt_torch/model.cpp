
#include <filesystem>

#include "tensorrt_torch/model.hpp"
#include "tensorrt_torch/allocator.hpp"

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

        nvinfer1::cudaPointerAttributes attributes;
        HAMI_ASSERT(nvinfer1::cudaPointerGetAttributes(
                        &attributes, destination) == nvinfer1::cudaSuccess);

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
            ASSERT(cudaMemcpyAsync(destination, tmpBuf.get(), nbBytes,
                                   cudaMemcpyHostToDevice,
                                   stream) == nvinfer1::cudaSuccess);
            // ok to free tmpBuf
            // cudaMemcpyAsync will return once the pageable buffer has been.
            return file_stream.gcount();
        }
        return -1;
    }

    void reset() {
        ASSERT(file_stream.good());
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

void LoadTensorrtFromFile::init(
    const std::unordered_map<std::string, std::string>& config,
    const hami::dict& dict_config) {
    HAMI_ASSERT(dict_config);
    if (dict_config->find(TASK_ENGINE_KEY) != dict_config->end()) {
        SPDLOG_INFO(
            "LoadTensorrtFromFile: aready loaded engine in dict_config, skip "
            "loading.");
        return;
    }
    // handle instance index
    // str::try_update(config, TASK_INDEX_KEY, instance_index_);
    // str::try_update(config, TASK_INDEX_KEY, instance_num_);
    // HAMI_ASSERT(instance_num_ >= 1 && instance_index_ >= 0);

    // initialize converter, get std::shared_ptr<ICudaEngine>
    HAMI_ASSERT(config.find("model") != config.end(),
                "`model` is not found in config");

    LocalFileStreamReader reader(config.at("model"));
    runtime_ = std::make_unique<nvinfer1::IRuntime>();
    allocator_ = std::make_unique<TorchAsyncAllocator>();
    runtime_->setGpuAllocator(allocator_.get());
    auto* engine_ptr = runtime_->deserializeCudaEngine(reader);
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(engine_ptr);

    (*dict_config)[TASK_ENGINE_KEY] = engine_;
}

void LoadTensorrtFromOnnx::init(
    const std::unordered_map<std::string, std::string>& config,
    const hami::dict& dict_config) {
    HAMI_ASSERT(dict_config);
    if (dict_config->find(TASK_ENGINE_KEY) != dict_config->end()) {
        SPDLOG_INFO(
            "LoadTensorrtFromFile: aready loaded engine in dict_config, skip "
            "loading.");
        return;
    }
    // handle instance index
    // str::try_update(config, TASK_INDEX_KEY, instance_index_);
    str::try_update(config, TASK_INDEX_KEY, instance_num_);
    // HAMI_ASSERT(instance_num_ >= 1 && instance_index_ >= 0);

    // set runtime && allocator
    runtime_ = std::make_unique<nvinfer1::IRuntime>();
    allocator_ = std::make_unique<TorchAsyncAllocator>();
    runtime_->setGpuAllocator(allocator_.get());

    // initialize converter, get std::shared_ptr<ICudaEngine>
    HAMI_ASSERT(config.find("model") != config.end(),
                "`model` is not found in config");

    OnnxParams params = config2onnxparams(config);

    auto mem = !hami::filesystem::exists(params.cache_path) ? onnx2trt(params)
                                                            : nullptr;

    if (mem) {
        auto* engine_ptr =
            runtime_->deserializeCudaEngine(mem->data(), mem->size());
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(engine_ptr);
    } else {
        LocalFileStreamReader reader(params.cache_path);
        auto* engine_ptr = runtime_->deserializeCudaEngine(reader);
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(engine_ptr);
    }

    (*dict_config)[TASK_ENGINE_KEY] = engine_;
}
}  // namespace torchpipe