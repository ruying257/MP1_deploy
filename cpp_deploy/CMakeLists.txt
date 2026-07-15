cmake_minimum_required(VERSION 3.16)
project(mp1_deploy_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

option(MP1_ENABLE_UR_RTDE "Enable UR RTDE command sending in mp1_real_robot_control" OFF)
option(MP1_ENABLE_TENSORRT "Enable TensorRT 8.x policy runtime" OFF)

add_library(mp1_deploy
    src/observation_buffer.cpp
    src/runtime_factory.cpp
    src/safety_filter.cpp
    src/torchscript_runtime.cpp
)
target_include_directories(mp1_deploy PUBLIC include)
target_link_libraries(mp1_deploy PUBLIC "${TORCH_LIBRARIES}")

if(MP1_ENABLE_TENSORRT)
    find_package(CUDAToolkit REQUIRED)
    find_package(nlohmann_json CONFIG REQUIRED)
    find_package(OpenSSL REQUIRED)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ENV TENSORRT_ROOT
        PATH_SUFFIXES include include/aarch64-linux-gnu include/x86_64-linux-gnu)
    find_library(TENSORRT_NVINFER_LIBRARY nvinfer HINTS ENV TENSORRT_ROOT
        PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu lib/x86_64-linux-gnu)
    find_library(TENSORRT_NVINFER_PLUGIN_LIBRARY nvinfer_plugin HINTS ENV TENSORRT_ROOT
        PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu lib/x86_64-linux-gnu)
    if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIBRARY OR NOT TENSORRT_NVINFER_PLUGIN_LIBRARY)
        message(FATAL_ERROR "TensorRT 8.x headers or libraries not found. Set TENSORRT_ROOT or install TensorRT development packages.")
    endif()
    file(READ "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" TENSORRT_VERSION_HEADER)
    string(REGEX MATCH "#define NV_TENSORRT_MAJOR +([0-9]+)" _ ${TENSORRT_VERSION_HEADER})
    if(NOT CMAKE_MATCH_1 STREQUAL "8")
        message(FATAL_ERROR "MP1 TensorRT runtime supports TensorRT 8.x binding API only.")
    endif()
    target_sources(mp1_deploy PRIVATE src/trt_runtime.cpp)
    target_include_directories(mp1_deploy PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(mp1_deploy PUBLIC CUDAToolkit::cudart nlohmann_json::nlohmann_json OpenSSL::Crypto
        ${TENSORRT_NVINFER_LIBRARY} ${TENSORRT_NVINFER_PLUGIN_LIBRARY})
    target_compile_definitions(mp1_deploy PUBLIC MP1_WITH_TENSORRT=1)
endif()

add_executable(mp1_offline_infer src/offline_infer.cpp)
target_link_libraries(mp1_offline_infer PRIVATE mp1_deploy)

add_executable(mp1_dry_run src/dry_run.cpp)
target_link_libraries(mp1_dry_run PRIVATE mp1_deploy)

add_executable(mp1_real_input_dry_run src/real_input_dry_run.cpp)
target_link_libraries(mp1_real_input_dry_run PRIVATE mp1_deploy)

add_executable(mp1_real_robot_control src/real_robot_control.cpp)
target_link_libraries(mp1_real_robot_control PRIVATE mp1_deploy)
if(MP1_ENABLE_UR_RTDE)
    find_package(ur_rtde REQUIRED)
    target_compile_definitions(mp1_real_robot_control PRIVATE MP1_WITH_UR_RTDE=1)
    if(TARGET ur_rtde::rtde)
        target_link_libraries(mp1_real_robot_control PRIVATE ur_rtde::rtde)
    elseif(TARGET ur_rtde)
        target_link_libraries(mp1_real_robot_control PRIVATE ur_rtde)
    else()
        target_link_libraries(mp1_real_robot_control PRIVATE ur_rtde)
    endif()
endif()

enable_testing()
add_executable(test_safety_filter tests/test_safety_filter.cpp)
target_link_libraries(test_safety_filter PRIVATE mp1_deploy)
add_test(NAME safety_filter COMMAND test_safety_filter)
if(MP1_ENABLE_TENSORRT)
    add_executable(test_trt_runtime_meta tests/test_trt_runtime_meta.cpp)
    target_link_libraries(test_trt_runtime_meta PRIVATE mp1_deploy)
    add_test(NAME trt_runtime_meta COMMAND test_trt_runtime_meta
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/trt_runtime_meta.json)
endif()
