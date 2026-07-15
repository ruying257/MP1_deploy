#include "mp1_deploy/trt_runtime.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>

#include <fstream>
#include <array>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mp1_deploy {
namespace {

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "TensorRT: " << message << "\n";
        }
    }
};

template <typename T>
struct TrtDeleter {
    void operator()(T* value) const noexcept {
        if (value != nullptr) {
            value->destroy();
        }
    }
};

using RuntimePtr = std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>>;
using EnginePtr = std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>>;
using ContextPtr = std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>>;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

bool has_dynamic_dimension(const nvinfer1::Dims& dims) {
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] < 0) {
            return true;
        }
    }
    return false;
}

std::vector<int64_t> dims_to_shape(const nvinfer1::Dims& dims, const std::string& binding_name) {
    std::vector<int64_t> shape;
    shape.reserve(dims.nbDims);
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] < 0) {
            throw std::runtime_error("TensorRT binding has unresolved dynamic dimension: " + binding_name);
        }
        shape.push_back(dims.d[index]);
    }
    return shape;
}

torch::ScalarType trt_to_torch_dtype(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return torch::kFloat32;
        case nvinfer1::DataType::kHALF:
            return torch::kFloat16;
        case nvinfer1::DataType::kINT8:
            return torch::kInt8;
        case nvinfer1::DataType::kINT32:
            return torch::kInt32;
        case nvinfer1::DataType::kBOOL:
            return torch::kBool;
        default:
            throw std::runtime_error("Unsupported TensorRT binding dtype");
    }
}

class TrtEngine {
public:
    explicit TrtEngine(const std::string& engine_path) {
        if (!initLibNvInferPlugins(&logger_, "")) {
            throw std::runtime_error("Failed to initialize TensorRT plugins");
        }
        std::ifstream input(engine_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("Failed to open TensorRT engine: " + engine_path);
        }
        const std::vector<char> serialized(
            (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        if (serialized.empty()) {
            throw std::runtime_error("TensorRT engine is empty: " + engine_path);
        }

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }
        engine_.reset(runtime_->deserializeCudaEngine(serialized.data(), serialized.size()));
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize TensorRT engine: " + engine_path);
        }
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            throw std::runtime_error("Failed to create TensorRT execution context: " + engine_path);
        }
        for (int index = 0; index < engine_->getNbBindings(); ++index) {
            binding_indices_.emplace(engine_->getBindingName(index), index);
        }
    }

    std::unordered_map<std::string, torch::Tensor> infer(
        const std::unordered_map<std::string, torch::Tensor>& inputs,
        cudaStream_t stream) {
        std::vector<void*> bindings(engine_->getNbBindings(), nullptr);
        std::unordered_map<std::string, torch::Tensor> buffers;

        for (const auto& [name, value] : inputs) {
            const int index = binding_index(name);
            if (!engine_->bindingIsInput(index)) {
                throw std::runtime_error("TensorRT binding is not an input: " + name);
            }
            const auto dtype = trt_to_torch_dtype(engine_->getBindingDataType(index));
            torch::Tensor tensor = value.to(torch::Device(torch::kCUDA), dtype, false, false).contiguous();
            const auto expected = engine_->getBindingDimensions(index);
            if (has_dynamic_dimension(expected)) {
                nvinfer1::Dims actual{};
                actual.nbDims = static_cast<int>(tensor.dim());
                for (int dim = 0; dim < actual.nbDims; ++dim) {
                    actual.d[dim] = static_cast<int>(tensor.size(dim));
                }
                if (!context_->setBindingDimensions(index, actual)) {
                    throw std::runtime_error("Failed to set TensorRT binding dimensions: " + name);
                }
            } else if (tensor.sizes().vec() != dims_to_shape(expected, name)) {
                throw std::runtime_error("TensorRT input shape mismatch: " + name);
            }
            buffers.emplace(name, tensor);
            bindings[index] = buffers.at(name).data_ptr();
        }

        if (!context_->allInputDimensionsSpecified()) {
            throw std::runtime_error("TensorRT input dimensions are not fully specified");
        }
        for (int index = 0; index < engine_->getNbBindings(); ++index) {
            if (engine_->bindingIsInput(index)) {
                if (bindings[index] == nullptr) {
                    throw std::runtime_error("TensorRT input binding was not supplied: "
                        + std::string(engine_->getBindingName(index)));
                }
                continue;
            }
            const std::string name = engine_->getBindingName(index);
            buffers.emplace(name, torch::empty(
                dims_to_shape(context_->getBindingDimensions(index), name),
                torch::TensorOptions()
                    .dtype(trt_to_torch_dtype(engine_->getBindingDataType(index)))
                    .device(torch::kCUDA)));
            bindings[index] = buffers.at(name).data_ptr();
        }

        if (!context_->enqueueV2(bindings.data(), stream, nullptr)) {
            throw std::runtime_error("TensorRT enqueueV2 failed");
        }

        std::unordered_map<std::string, torch::Tensor> outputs;
        for (int index = 0; index < engine_->getNbBindings(); ++index) {
            if (!engine_->bindingIsInput(index)) {
                const std::string name = engine_->getBindingName(index);
                outputs.emplace(name, buffers.at(name));
            }
        }
        return outputs;
    }

private:
    int binding_index(const std::string& name) const {
        const auto it = binding_indices_.find(name);
        if (it == binding_indices_.end()) {
            throw std::runtime_error("TensorRT binding not found: " + name);
        }
        return it->second;
    }

    TrtLogger logger_;
    RuntimePtr runtime_;
    EnginePtr engine_;
    ContextPtr context_;
    std::unordered_map<std::string, int> binding_indices_;
};

std::vector<float> json_float_vector(const nlohmann::json& value, const char* name) {
    if (!value.is_array()) {
        throw std::runtime_error(std::string("TensorRT metadata field is not an array: ") + name);
    }
    return value.get<std::vector<float>>();
}

}  // namespace

TrtRuntimeMeta load_trt_runtime_meta(const std::string& metadata_path) {
    std::ifstream input(metadata_path);
    if (!input) {
        throw std::runtime_error("Failed to open TensorRT runtime metadata: " + metadata_path);
    }
    const nlohmann::json json = nlohmann::json::parse(input);
    TrtRuntimeMeta meta;
    meta.image_input_dtype = json.at("image_input_dtype").get<std::string>();
    meta.n_obs_steps = json.at("n_obs_steps").get<int>();
    meta.horizon = json.at("horizon").get<int>();
    meta.action_dim = json.at("action_dim").get<int>();
    meta.n_action_steps = json.at("n_action_steps").get<int>();
    meta.num_inference_steps = json.at("num_inference_steps").get<int>();
    meta.dt = json.at("dt").get<float>();
    const auto& normalizer = json.at("action_normalizer");
    meta.action_scale = json_float_vector(normalizer.at("scale"), "action_normalizer.scale");
    meta.action_offset = json_float_vector(normalizer.at("offset"), "action_normalizer.offset");
    if ((meta.image_input_dtype != "float32" && meta.image_input_dtype != "uint8")
        || meta.n_obs_steps <= 0 || meta.horizon <= 0 || meta.action_dim <= 0
        || meta.n_action_steps <= 0 || meta.num_inference_steps <= 0 || meta.dt <= 0.0F
        || meta.n_obs_steps > meta.horizon
        || meta.n_obs_steps - 1 + meta.n_action_steps > meta.horizon
        || static_cast<int>(meta.action_scale.size()) != meta.action_dim
        || static_cast<int>(meta.action_offset.size()) != meta.action_dim) {
        throw std::runtime_error("TensorRT runtime metadata has an invalid action contract");
    }
    return meta;
}

std::string trt_file_sha256(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open file for SHA256: " + path);
    }
    SHA256_CTX context;
    SHA256_Init(&context);
    std::array<unsigned char, 8192> buffer{};
    while (input.good()) {
        input.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        const auto count = input.gcount();
        if (count > 0) {
            SHA256_Update(&context, buffer.data(), static_cast<size_t>(count));
        }
    }
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    SHA256_Final(digest.data(), &context);
    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(digest.size() * 2);
    for (const unsigned char byte : digest) {
        result.push_back(hex[(byte >> 4) & 0x0F]);
        result.push_back(hex[byte & 0x0F]);
    }
    return result;
}

class TrtPolicyRuntime::Impl {
public:
    explicit Impl(const TrtRuntimeOptions& options)
        : meta(load_trt_runtime_meta(options.metadata_path)),
          obs_engine(options.obs_engine_path),
          unet_engine(options.unet_engine_path),
          action_scale(torch::tensor(meta.action_scale,
              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))),
          action_offset(torch::tensor(meta.action_offset,
              torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))) {}

    TrtRuntimeMeta meta;
    TrtEngine obs_engine;
    TrtEngine unet_engine;
    torch::Tensor action_scale;
    torch::Tensor action_offset;
};

TrtPolicyRuntime::TrtPolicyRuntime(TrtRuntimeOptions options)
    : impl_(std::make_unique<Impl>(options)) {}

std::pair<torch::Tensor, torch::Tensor> TrtPolicyRuntime::infer(const PolicyInputs& inputs) {
    const cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();
    const auto obs_outputs = impl_->obs_engine.infer(
        {
            {"global_image", inputs.global_image},
            {"wrist_image", inputs.wrist_image},
            {"point_cloud", inputs.point_cloud},
            {"agent_pos", inputs.agent_pos},
        },
        stream);
    const auto global_cond_it = obs_outputs.find("global_cond");
    if (global_cond_it == obs_outputs.end()) {
        throw std::runtime_error("obs_encoder engine did not produce global_cond");
    }
    const torch::Tensor global_cond = global_cond_it->second;
    torch::Tensor x_current = inputs.initial_noise
        .to(torch::Device(torch::kCUDA), torch::kFloat32, false, false).contiguous();
    if (x_current.dim() != 3 || x_current.size(1) != impl_->meta.horizon
        || x_current.size(2) != impl_->meta.action_dim) {
        throw std::runtime_error("initial_noise does not match TensorRT runtime metadata");
    }
    const torch::Tensor r = x_current.select(1, 0).select(1, 0) * 0.0F;
    for (int index = 0; index < impl_->meta.num_inference_steps; ++index) {
        const torch::Tensor timestep = r + static_cast<float>(index) / impl_->meta.num_inference_steps;
        const auto unet_outputs = impl_->unet_engine.infer(
            {{"x_current", x_current}, {"timestep", timestep}, {"global_cond", global_cond}, {"r", r}},
            stream);
        const auto v_pred_it = unet_outputs.find("v_pred");
        if (v_pred_it == unet_outputs.end()) {
            throw std::runtime_error("unet_step engine did not produce v_pred");
        }
        x_current = x_current + v_pred_it->second.to(torch::kFloat32) * impl_->meta.dt;
    }

    const torch::Tensor normalized = x_current.slice(2, 0, impl_->meta.action_dim);
    const torch::Tensor action_pred = ((normalized.reshape({-1, impl_->meta.action_dim}) - impl_->action_offset)
        / impl_->action_scale).reshape(normalized.sizes());
    const int start = impl_->meta.n_obs_steps - 1;
    const torch::Tensor action = action_pred.slice(1, start, start + impl_->meta.n_action_steps);
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    return {action.cpu(), action_pred.cpu()};
}

}  // namespace mp1_deploy
