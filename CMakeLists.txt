cmake_minimum_required(VERSION 3.10.0)

project(test_cublaslt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${PROJECT_NAME}
    test_data_order.cu
    gemmInt8.cu
    gemmFp16.cu
    cublas.cu
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(CUDART_LIBRARY cudart)
set(CUBLAS_LIBRARY cublas)
set(CUBLASLT_LIBRARY cublasLt)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBLAS_LIBRARY cublas ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBLASLT_LIBRARY cublasLt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

target_link_libraries(${PROJECT_NAME}
    ${CUDART_LIBRARY}
    ${CUBLASLT_LIBRARY}
    ${CUBLAS_LIBRARY}
)