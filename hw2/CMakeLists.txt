cmake_minimum_required(VERSION 3.18)
project(TensorProject LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architectures (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86)

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Create Tensor library
add_library(Tensor STATIC
    Tensor.cu
    Tensor.h
)

# Set properties for Tensor library
set_target_properties(Tensor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Create ActivationFunctions library
add_library(ActivationFunctions STATIC
    ActivationFunctions.cu
    ActivationFunctions.h
)

# Set properties for ActivationFunctions library
set_target_properties(ActivationFunctions PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link ActivationFunctions with Tensor
target_link_libraries(ActivationFunctions Tensor)

# Create test executables
add_executable(test_tensor test_tensor.cu)
target_link_libraries(test_tensor Tensor)

add_executable(test_activations test_activations.cu)
target_link_libraries(test_activations ActivationFunctions Tensor)

# Set properties for test executables
set_target_properties(test_tensor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(test_activations PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable testing
enable_testing()

# Add tests
add_test(NAME TensorTests COMMAND test_tensor)
add_test(NAME ActivationTests COMMAND test_activations)

# Custom targets for convenience
add_custom_target(run_tensor_tests
    COMMAND test_tensor
    DEPENDS test_tensor
    COMMENT "Running Tensor tests"
)

add_custom_target(run_activation_tests
    COMMAND test_activations
    DEPENDS test_activations
    COMMENT "Running Activation Function tests"
)

add_custom_target(run_all_tests
    COMMAND test_tensor && test_activations
    DEPENDS test_tensor test_activations
    COMMENT "Running all tests"
)

# Debug configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(Tensor PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-g -G>
        $<$<COMPILE_LANGUAGE:CXX>:-g>
    )
    target_compile_options(ActivationFunctions PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-g -G>
        $<$<COMPILE_LANGUAGE:CXX>:-g>
    )
endif()

# Print build information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA standard: ${CMAKE_CUDA_STANDARD}")