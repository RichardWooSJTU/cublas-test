cmake_minimum_required(VERSION 3.10.0)

project(test_cublaslt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

LINK_DIRECTORIES(/usr/local/cuda/lib64)

add_executable(${PROJECT_NAME}
    main.cc
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(CUDART_LIBRARY cudart)
set(CUBLAS_LIBRARY cublas)
set(CUBLASLT_LIBRARY cublasLt)

find_library(CUDART_LIBRARY cudart)
find_library(CUBLAS_LIBRARY cublas)
find_library(CUBLASLT_LIBRARY cublasLt)

target_link_libraries(${PROJECT_NAME}
    ${CUDART_LIBRARY}
    ${CUBLASLT_LIBRARY}
    ${CUBLAS_LIBRARY}
)