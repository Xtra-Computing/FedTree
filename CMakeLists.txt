cmake_minimum_required(VERSION 3.15)
#cmake_policy(SET CMP0104 NEW)
#cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0042 NEW)
project(FedTree LANGUAGES C CXX)

if(MSVC)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif()

option(BUILD_SHARED_LIBS "Build as a shared library" ON)
option(USE_CUDA "Compile with CUDA for homomorphic encryption" OFF)
option(USE_CUDA_ARRAY "Compile with CUDA for training" OFF)
option(DISTRIBUTED "Build for distributed computing" ON)
option(USE_DOUBLE "Use double as gradient_type" OFF)
set(BUILD_TESTS OFF CACHE BOOL "Build Tests")
set(NTL_PATH "/usr/local" CACHE STRING "NTL Path")

find_package(OpenMP REQUIRED)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
if (MSVC AND BUILD_SHARED_LIBS)
    set (CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif ()

if (USE_CUDA)
    message("Compile with CUDA")
    find_package(CUDA REQUIRED QUIET)

    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -lineinfo --expt-extended-lambda --default-stream per-thread")
    include_directories(${PROJECT_SOURCE_DIR}/CGBN/include)
    add_subdirectory(${PROJECT_SOURCE_DIR}/thrust/)
    find_package(Thrust REQUIRED)
    thrust_create_target(ThrustOMP HOST CPP DEVICE OMP)
    include_directories(${NTL_PATH}/include/)
else ()
    message("Complie without CUDA")
    #set(Thrust_DIR "${PROJECT_SOURCE_DIR}/thrust/cmake/")
    add_subdirectory(${PROJECT_SOURCE_DIR}/thrust/)
    find_package(Thrust REQUIRED)
    thrust_create_target(ThrustOMP HOST CPP DEVICE OMP)
#    include_directories(/usr/local/include/)
    include_directories(${NTL_PATH}/include/)
endif ()

# Require the users to install gRPC in prior.
find_package(gRPC CONFIG REQUIRED)

#if (DISTRIBUTED)
#    include(FetchContent)
#    FetchContent_Declare(
#            gRPC
#            GIT_REPOSITORY https://github.com/grpc/grpc
#            GIT_TAG        v1.35.0  # e.g v1.28.0
#    )
#    set(FETCHCONTENT_QUIET OFF)
#    FetchContent_MakeAvailable(gRPC)
#endif ()

if (CMAKE_VERSION VERSION_LESS "3.1")
    add_compile_options("-std=c++11")
else ()
    set(CMAKE_CXX_STANDARD 11)
endif ()

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

# for easylogging++ configuration
add_definitions("-DELPP_FEATURE_PERFORMANCE_TRACKING")
add_definitions("-DELPP_THREAD_SAFE")
add_definitions("-DELPP_STL_LOGGING")
add_definitions("-DELPP_NO_LOG_TO_FILE")



# includes
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
if(USE_CUDA)
    list(REMOVE_ITEM COMMON_INCLUDES "${PROJECT_SOURCE_DIR}/include/FedTree/Encryption/paillier.h")
else()
    list(REMOVE_ITEM COMMON_INCLUDES "${PROJECT_SOURCE_DIR}/include/FedTree/Encryption/paillier_gpu.h")
endif()


set(DATASET_DIR ${PROJECT_SOURCE_DIR}/dataset/)
configure_file(include/FedTree/config.h.in config.h)

include_directories(${COMMON_INCLUDES})


if (USE_CUDA)
    include_directories(${PROJECT_SOURCE_DIR}/cub)
endif ()
add_subdirectory(${PROJECT_SOURCE_DIR}/src/FedTree)

if (BUILD_TESTS)
    message("Building tests")
    add_subdirectory(src/test)
endif ()

# configuration file

