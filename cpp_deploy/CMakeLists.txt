cmake_minimum_required(VERSION 3.16)
project(mp1_deploy_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)

option(MP1_ENABLE_UR_RTDE "Enable UR RTDE command sending in mp1_real_robot_control" OFF)

add_library(mp1_deploy
    src/observation_buffer.cpp
    src/safety_filter.cpp
    src/torchscript_runtime.cpp
)
target_include_directories(mp1_deploy PUBLIC include)
target_link_libraries(mp1_deploy PUBLIC "${TORCH_LIBRARIES}")

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
