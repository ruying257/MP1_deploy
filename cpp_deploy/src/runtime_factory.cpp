#include "mp1_deploy/policy_runtime.hpp"

#ifdef MP1_WITH_TENSORRT
#include "mp1_deploy/trt_runtime.hpp"
#endif

#include <stdexcept>

namespace mp1_deploy {

std::unique_ptr<PolicyRuntime> create_policy_runtime(
    const std::string& backend,
    const std::string& model_path,
    const torch::Device& device,
    const TrtRuntimeOptions& trt_options) {
    if (backend == "torchscript") {
        if (model_path.empty()) {
            throw std::runtime_error("--model is required when --backend torchscript");
        }
        return std::make_unique<TorchScriptRuntime>(model_path, device);
    }
    if (backend == "tensorrt") {
#ifdef MP1_WITH_TENSORRT
        if (!device.is_cuda()) {
            throw std::runtime_error("TensorRT backend requires --device cuda");
        }
        if (trt_options.obs_engine_path.empty() || trt_options.unet_engine_path.empty()
            || trt_options.metadata_path.empty()) {
            throw std::runtime_error("TensorRT backend requires --obs-engine, --unet-engine, and --trt-meta");
        }
        return std::make_unique<TrtPolicyRuntime>(trt_options);
#else
        throw std::runtime_error("TensorRT backend was not built. Reconfigure with -DMP1_ENABLE_TENSORRT=ON");
#endif
    }
    throw std::runtime_error("Unsupported --backend: " + backend + "; expected torchscript or tensorrt");
}

}  // namespace mp1_deploy
