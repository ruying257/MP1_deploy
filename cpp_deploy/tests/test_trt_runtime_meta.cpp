#include "mp1_deploy/trt_runtime.hpp"

#include <cassert>
#include <iostream>

int main(int argc, char** argv) {
    assert(argc == 2);
    const auto meta = mp1_deploy::load_trt_runtime_meta(argv[1]);
    assert(meta.image_input_dtype == "float32");
    assert(meta.n_obs_steps == 2);
    assert(meta.horizon == 4);
    assert(meta.action_dim == 7);
    assert(meta.n_action_steps == 3);
    assert(meta.num_inference_steps == 1);
    assert(meta.action_scale.size() == 7);
    assert(meta.action_offset.size() == 7);
    std::cout << "TensorRT runtime metadata test passed\n";
    return 0;
}
