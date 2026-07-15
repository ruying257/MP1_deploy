#pragma once

#include "mp1_deploy/policy_runtime.hpp"

#include <string>
#include <vector>

namespace mp1_deploy {

struct TrtRuntimeMeta {
    std::string image_input_dtype;
    int n_obs_steps = 0;
    int horizon = 0;
    int action_dim = 0;
    int n_action_steps = 0;
    int num_inference_steps = 0;
    float dt = 0.0F;
    std::vector<float> action_scale;
    std::vector<float> action_offset;
};

TrtRuntimeMeta load_trt_runtime_meta(const std::string& metadata_path);
std::string trt_file_sha256(const std::string& path);

class TrtPolicyRuntime final : public PolicyRuntime {
public:
    explicit TrtPolicyRuntime(TrtRuntimeOptions options);
    std::pair<torch::Tensor, torch::Tensor> infer(const PolicyInputs& inputs) override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mp1_deploy
