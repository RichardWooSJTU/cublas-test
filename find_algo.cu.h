#include <algorithm>
#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <cublas_v2.h>

#pragma once
#include <iostream>
#include <vector>
#include <fstream>

#include "find_algo.cu.h"

typedef struct {
    cublasLtMatmulAlgo_t algo;
    cublasStatus_t status;
    float time;
    size_t workspaceSize;  // actual memory workspace needed
    cublasMath_t mathMode;
    cublasLtReductionScheme_t reductionScheme;
    int customOption;
    float wavesCount;
} customMatmulPerf_t;

typedef struct {
   cublasLtMatmulAlgo_t algo;
   int m;
   int n;
   int k;
   int algoId;
   int swizzle;
   int customOption;
   int tile;
   int splitK_val;
   int reductionScheme;
   int stages;
   float time;
} algoSelect_t;




const int splitKSequenceA[] = {2, 3, 4, 5, 6, 8, 12, 16, 32};

static inline bool time_compare_perf(const customMatmulPerf_t& perf_a, const customMatmulPerf_t& perf_b)
{
    return ((perf_a.status == CUBLAS_STATUS_SUCCESS) && (perf_a.time < perf_b.time));
}

static inline bool time_compare_algo_para(const algoSelect_t& algo_para_a, const algoSelect_t& algo_para_b)
{
    return (algo_para_a.time < algo_para_b.time);
}

template<typename InT, typename OutT>
static cublasStatus_t TestMatmulRun(cublasLtHandle_t ltHandle,
                                    cublasLtMatmulDesc_t matmulDesc,
                                    cublasLtMatrixLayout_t A_desc,
                                    cublasLtMatrixLayout_t B_desc,
                                    cublasLtMatrixLayout_t C_desc,
                                    const  InT* A,
                                    const  InT* B,
                                     OutT* C,
                                    const cublasLtMatmulAlgo_t& algo,
                                    customMatmulPerf_t& perfResults,
                                    cudaEvent_t& startEvent,
                                    cudaEvent_t& stopEvent
                                )
{
    cudaStream_t stream = 0;
    cublasLtMatmulHeuristicResult_t heurResult;
    cublasStatus_t algoStatus =
        cublasLtMatmulAlgoCheck(ltHandle, matmulDesc, A_desc, B_desc, C_desc, C_desc, &algo, &heurResult);
    if (algoStatus == CUBLAS_STATUS_SUCCESS) {
        cudaError_t err;
        err = cudaEventRecord(startEvent, stream);
         OutT alpha = 1, beta = 0;
        void* workSpace;
        cudaMalloc(&workSpace, heurResult.workspaceSize);
        int repeats = 10000;
        for (int loop = 0; loop < repeats; loop++) {
            cublasStatus_t oneRunStatus = cublasLtMatmul(ltHandle,
                                                            matmulDesc,
                                                            &alpha,
                                                            A,
                                                            A_desc,
                                                            B,
                                                            B_desc,
                                                            &beta,
                                                            C,
                                                            C_desc,
                                                            C,
                                                            C_desc,
                                                            &algo,
                                                            workSpace,
                                                            heurResult.workspaceSize,
                                                            stream);
            if (oneRunStatus != CUBLAS_STATUS_SUCCESS) {
                algoStatus = oneRunStatus;
                std::cout << "heurResult.workspaceSize " << heurResult.workspaceSize << std::endl;
                std::cout << "cublasLtMatmul error " << oneRunStatus << std::endl;
                break;
            }
            err = cudaDeviceSynchronize();
             if (err != cudaSuccess) {
                 printf("GGGGGG!!!!!");
             }
        }
        
        err = cudaEventRecord(stopEvent, stream);
        err = cudaEventSynchronize(stopEvent);
        
        float time;
        err = cudaEventElapsedTime(&time, startEvent, stopEvent);
        if (err != cudaSuccess) {
            algoStatus = CUBLAS_STATUS_INTERNAL_ERROR;
        }
        if (algoStatus == CUBLAS_STATUS_SUCCESS) {
            perfResults.algo = algo;
            perfResults.time = time / repeats;
            perfResults.workspaceSize = heurResult.workspaceSize;
            perfResults.wavesCount = heurResult.wavesCount;
        }
        cudaFree(workSpace);
    } else {
        printf("not enough workspace! %ld\n", heurResult.workspaceSize);
        algoStatus = CUBLAS_STATUS_NOT_SUPPORTED;  // Not enough workspace
    }
    return algoStatus;
}

template<typename InT, typename OutT>
std::vector<customMatmulPerf_t> FindAlgo(cublasLtHandle_t ltHandle,
             int m,
             int n,
             int k,
             const  InT* A,
             const  InT* B,
              OutT* C,
             cublasLtMatmulDesc_t matmulDesc,
             cublasLtMatrixLayout_t A_desc,
             cublasLtMatrixLayout_t B_desc,
             cublasLtMatrixLayout_t C_desc,
             cublasComputeType_t computeType,
             cudaDataType_t scaleType,
             cudaDataType_t Atype,
             cudaDataType_t Btype,
             cudaDataType_t Ctype
             ) {
    // Get Ids    
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoGetIds
    // Input
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    // Output
    int algoIdA[100];
    int nbAlgoIds;
    status = cublasLtMatmulAlgoGetIds(
        ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, 100, algoIdA, &nbAlgoIds);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << " cublasLtMatmulAlgoGetIds A GG with status " << status << std::endl;
    }

    // std::cout << "get " << nbAlgoIds << " algoIds" << std::endl;

    int AlgoCount = 0;
    int AlgoCombinations = 20000;
    // cublasLtMatmulAlgo_t algos[AlgoCombinations]; 
    std::vector<algoSelect_t> algos;
    // Loop over the Algo IDs
    for (int idx = 0; idx < nbAlgoIds; idx++) {
        // std::cout << "Process algo ID " << algoIdA[idx] << std::endl;
        cublasLtMatmulAlgo_t algo;
        
        /* Initialize algo structure with given Algp ID */
        // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoInit
        status =
            cublasLtMatmulAlgoInit(ltHandle, computeType, scaleType, Atype, Btype, Ctype, Ctype, algoIdA[idx], &algo);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cout << " cublasLtMatmulAlgoInit GG with status " << status << std::endl;
        }

        // Query the tiles enums supported by that algo which is used to alloc enough space to store it
        // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoCapGetAttribute
        size_t sizeWritten = 0;
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, NULL, 0, &sizeWritten);
        int nbTiles = int(sizeWritten / sizeof(int));
        std::vector<int> tileA(nbTiles == 0 ? 1 : nbTiles);
        if (nbTiles == 0) {
            tileA[0] = CUBLASLT_MATMUL_TILE_UNDEFINED;
            nbTiles = 1;
            // std::cout << "no tiles" << std::endl;
        } else {
            cublasLtMatmulAlgoCapGetAttribute(
                &algo, CUBLASLT_ALGO_CAP_TILE_IDS, tileA.data(), sizeof(int) * nbTiles, &sizeWritten);
        }
        // std::cout << "has tiles " << nbTiles << std::endl;
        // Query the stages enums supported by that algo (cuda must >= 11.0)
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, NULL, 0, &sizeWritten);
        int nbStages = int(sizeWritten / sizeof(int));
        std::vector<int> stagesA(nbStages == 0 ? 1 : nbStages);
        if (nbStages == 0) {
            stagesA[0] = CUBLASLT_MATMUL_STAGES_UNDEFINED;
            nbStages = 1;
            // std::cout << "no stages" << std::endl;
        }
        else {
            cublasLtMatmulAlgoCapGetAttribute(
                &algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stagesA.data(), sizeof(int) * nbStages, &sizeWritten);
                
        }

        // std::cout << "has stages " << nbStages << std::endl;
        // Retrieve Other Algo Capabilities attributes
        int splitkSupport, redMask, swizzlingMax, customOptionMax;
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitkSupport, sizeof(splitkSupport), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redMask, sizeof(redMask), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &swizzlingMax, sizeof(swizzlingMax), &sizeWritten);
        cublasLtMatmulAlgoCapGetAttribute(
            &algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &customOptionMax, sizeof(customOptionMax), &sizeWritten);

        /* Loop over the different tiles */
        for (int tileIdx = 0; tileIdx < nbTiles && AlgoCount < AlgoCombinations; tileIdx++) {
            /* Loop over different stages count */
            for (int stagesIdx = 0; stagesIdx < nbStages && AlgoCount < AlgoCombinations; stagesIdx++) {
                /* Loop over the different custom option if any */
                for (int customOption = 0; customOption <= customOptionMax && AlgoCount < AlgoCombinations; customOption++) {
                     /* Loop over the CTAs swizzling support */
                     for (int k = 0; k <= swizzlingMax && AlgoCount < AlgoCombinations; k++) {
                        int splitK_trial = 0;
                        if (splitkSupport) {
                            splitK_trial += sizeof(splitKSequenceA) / sizeof(splitKSequenceA[0]);
                        }

                        for (int l = 0; (l < (1 + splitK_trial)) && (AlgoCount < AlgoCombinations); l++) {
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileA[tileIdx], sizeof(tileA[tileIdx]));
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stagesA[stagesIdx], sizeof(stagesA[stagesIdx]));
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption, sizeof(customOption));
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &k, sizeof(k));
                            int splitK_val = 0;
                            int redScheme = CUBLASLT_REDUCTION_SCHEME_NONE;
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK_val, sizeof(splitK_val));
                            cublasLtMatmulAlgoConfigSetAttribute(
                                &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &redScheme, sizeof(int));
                            if (l > 0) {  // Split-K case
                                splitK_val = splitKSequenceA[l - 1];
                                cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                                                     CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                                                     &splitKSequenceA[l - 1],
                                                                     sizeof(splitKSequenceA[l - 1]));
                                for (redScheme = 1;
                                    redScheme < (int)CUBLASLT_REDUCTION_SCHEME_MASK && (AlgoCount < AlgoCombinations);
                                    redScheme = redScheme << 1) {
                                   if (redScheme & redMask) {
                                       cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                                                            CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                                            &redScheme,
                                                                            sizeof(redScheme));

                                       cublasLtMatmulHeuristicResult_t heurResult;
                                       cublasStatus_t algoStatus = cublasLtMatmulAlgoCheck(
                                        ltHandle, matmulDesc, A_desc, B_desc, C_desc, C_desc, &algo, &heurResult);
                                       if (algoStatus == CUBLAS_STATUS_SUCCESS) {
                                        // std::cout << "algo " << algoIdA[idx] << " tile " << tileA[tileIdx] << " stages " << stagesA[stagesIdx]
                                        //     <<  " customOption "  << customOption << " k " << k  << " l " << l << " redScheme " << redScheme  << std::endl;
                                        //    algos[AlgoCount++] = algo;
                                           algoSelect_t algoSelect;
                                           algoSelect.algo = algo;
                                           algoSelect.m = m;
                                           algoSelect.n = n;
                                           algoSelect.k = k;
                                           algoSelect.algoId = algoIdA[idx];
                                           algoSelect.tile = tileA[tileIdx];
                                           algoSelect.swizzle = k;
                                           algoSelect.customOption = customOption;
                                           algoSelect.splitK_val = splitK_val;
                                           algoSelect.reductionScheme = redScheme;
                                           algoSelect.stages = stagesA[stagesIdx];
                                           algos.push_back(algoSelect);
                                           AlgoCount++;

                                       }
                                   }  // end if
                               }                    
                            } else {
                                // Prepare algos
                                cublasLtMatmulHeuristicResult_t heurResult;
                                // https://docs.nvidia.com/cuda/cublas/index.html#cublasLtMatmulAlgoCheck
                                cublasStatus_t algoStatus = cublasLtMatmulAlgoCheck(
                                    ltHandle, matmulDesc, A_desc, B_desc, C_desc, C_desc, &algo, &heurResult);
                                if (algoStatus == CUBLAS_STATUS_SUCCESS) {
                                    std::cout << "algo " << algoIdA[idx] << " tile " << tileA[tileIdx] << " stages " << stagesA[stagesIdx]
                                        <<  " customOption "  << customOption << " k " << k    << std::endl;
                                    // algos[AlgoCount++] = algo;
                                    algoSelect_t algoSelect;
                                    algoSelect.algo = algo;
                                    algoSelect.m = m;
                                    algoSelect.n = n;
                                    algoSelect.k = k;
                                    algoSelect.algoId = algoIdA[idx];
                                    algoSelect.tile = tileA[tileIdx];
                                    algoSelect.swizzle = k;
                                    algoSelect.customOption = customOption;
                                    algoSelect.splitK_val = splitK_val;
                                    algoSelect.reductionScheme = redScheme;
                                    algoSelect.stages = stagesA[stagesIdx];
                                    algos.push_back(algoSelect);
                                    AlgoCount++;
                                }
                            }
                            
                        }
                     }
                }
            }
        }
    }
    std::cout << "Got " << AlgoCount << " algos"  << std::endl;
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    std::vector<customMatmulPerf_t> perfResults(AlgoCount);
    if (cudaEventCreate(&startEvent, cudaEventBlockingSync) != cudaSuccess) {
        std::cout << " cudaEventCreate GG with status " << std::endl;
    }
    if (cudaEventCreate(&stopEvent, cudaEventBlockingSync) != cudaSuccess) {
        std::cout << " cudaEventCreate GG with status "<< std::endl;
    }
    for (int i = 0; i < AlgoCount; i++) {
        status = TestMatmulRun(ltHandle,
            matmulDesc,
            A_desc,
            B_desc,
            C_desc,
            A,
            B,
            C,
            algos[i].algo,
            perfResults[i],
            startEvent,
            stopEvent);
        perfResults[i].status = status;
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cout << "GG TestMatmulRun with status " << status << std::endl;
        }
        algos[i].time = perfResults[i].time;
        // std::cout << "algo "<< i << " time " << perfResults[i].time << " wavesCount " << perfResults[i].wavesCount << " workspaceSize " << perfResults[i].workspaceSize << std::endl;


    }
    std::sort(algos.begin(), algos.end(), time_compare_algo_para);
    int i = 0;
    while (algos[i].time == 0) i++;
    // return perfResults;
    std::ofstream outfile;
    outfile.open("select.csv", std::ios::app);
    outfile << m << ", "  << k << ", " << n << ", " << algos[i].algoId << ", " <<
        algos[i].swizzle << ", " << algos[i].customOption << ", " << algos[i].tile << ", " << algos[i].splitK_val << 
        ", " << algos[i].reductionScheme << ", " << algos[i].stages << ", " <<  algos[i].time  << std::endl;
    outfile.close();
    return perfResults;
    // std::cout << "fast algo " << " time " << perfResults[i].time << " wavesCount " << perfResults[i].wavesCount << " workspaceSize " << perfResults[i].workspaceSize << std::endl;
}