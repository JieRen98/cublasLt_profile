#include <cuda_runtime_api.h>
#include <cublasLt.h>

#include <iostream>

namespace mixed_kernels {
#define CHECK_CUBLAS(statement)                        \
  do {                                                 \
    auto status = statement;                           \
    if (status != CUBLAS_STATUS_SUCCESS) {             \
      printf("Failed at %s:%d\n", __FILE__, __LINE__); \
    }                                                  \
                                                       \
  } while (0)

#define CHECK_CUDA(statement)                        \
  do {                                                 \
    auto status = statement;                           \
    if (status != cudaSuccess) {             \
      printf("Failed at %s:%d\n", __FILE__, __LINE__); \
    }                                                  \
                                                       \
  } while (0)

    namespace cublasLtFp8RowMajorNTNMeta {
        namespace {
            struct Impl {
                constexpr static int searchAlgoNum = 10;
                const uint64_t m, n, k;
                const int64_t lda, ldb, ldc;
                const std::size_t workspaceSize;
                void *workspace;
                cublasLtHandle_t handle{nullptr};
                const cublasOperation_t transa{CUBLAS_OP_T}, transb{CUBLAS_OP_N};
                cublasLtMatmulPreference_t preference{nullptr};
                cublasLtMatrixLayout_t Adesc{nullptr}, Bdesc{nullptr};
                cublasLtMatmulDesc_t operationDesc{nullptr};
                struct {
                    cublasLtMatrixLayout_t Cdesc{nullptr}, Ddesc{nullptr};
                    cublasLtMatmulHeuristicResult_t heuristicResult[searchAlgoNum];
                    int algoNum = searchAlgoNum;
                } fp8fp8fp16{}, fp8fp8fp32{};
            };
        }  // namespace

        CUBLASAPI cublasStatus_t CUBLASWINAPI
        create(void **instance, uint64_t m, uint64_t n, uint64_t k, int64_t lda,
               int64_t ldb, int64_t ldc, std::size_t workspaceSize, void *workspace);

        CUBLASAPI cublasStatus_t CUBLASWINAPI destroy(void *instance);

        static CUBLASAPI cublasStatus_t CUBLASWINAPI
        matmul(void *instance, cudaStream_t stream, double alpha, const void *A,
               const void *B, double beta, void *C, cudaDataType Ctype, int algoIdx = 0);

        static CUBLASAPI cublasStatus_t CUBLASWINAPI
        getAlgoNum(void *instance, cudaStream_t stream, double alpha, const void *A,
                   const void *B, double beta, void *C, cudaDataType Ctype, int *algoNum);
    }  // namespace cublasLtFp8RowMajorNTNMeta

    cublasStatus_t cublasLtFp8RowMajorNTNMeta::create(
            void **instance, uint64_t m, uint64_t n, uint64_t k, int64_t lda,
            int64_t ldb, int64_t ldc, std::size_t workspaceSize, void *workspace) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties_v2(&deviceProp, 0);
        if (deviceProp.major < 9) {
            return CUBLAS_STATUS_SUCCESS;
        }
        auto impl = new Impl{m, n, k, lda, ldb, ldc, workspaceSize, workspace};
        CHECK_CUBLAS(cublasLtCreate(&impl->handle));
        CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&impl->preference));
        CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
                impl->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspaceSize, sizeof(workspaceSize)));
        CHECK_CUBLAS(
                cublasLtMatrixLayoutCreate(&impl->Adesc, CUDA_R_8F_E4M3, k, m, lda));
        CHECK_CUBLAS(
                cublasLtMatrixLayoutCreate(&impl->Bdesc, CUDA_R_8F_E4M3, k, n, ldb));
        CHECK_CUBLAS(cublasLtMatmulDescCreate(&impl->operationDesc,
                                              CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
                impl->operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &impl->transa,
                sizeof(impl->transa)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
                impl->operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &impl->transb,
                sizeof(impl->transb)));

        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&impl->fp8fp8fp16.Cdesc, CUDA_R_16F,
                                                n, m, ldc));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&impl->fp8fp8fp16.Ddesc, CUDA_R_16F,
                                                n, m, ldc));
        CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
                impl->handle, impl->operationDesc, impl->Bdesc, impl->Adesc,
                impl->fp8fp8fp16.Cdesc, impl->fp8fp8fp16.Ddesc, impl->preference, impl->fp8fp8fp16.algoNum,
                impl->fp8fp8fp16.heuristicResult, &impl->fp8fp8fp16.algoNum));
        if (impl->fp8fp8fp16.algoNum == 0) {
            CHECK_CUBLAS(CUBLAS_STATUS_NOT_SUPPORTED);
        }

        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&impl->fp8fp8fp32.Cdesc, CUDA_R_32F,
                                                n, m, ldc));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&impl->fp8fp8fp32.Ddesc, CUDA_R_32F,
                                                n, m, ldc));
        CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
                impl->handle, impl->operationDesc, impl->Bdesc, impl->Adesc,
                impl->fp8fp8fp32.Cdesc, impl->fp8fp8fp32.Ddesc, impl->preference, impl->fp8fp8fp32.algoNum,
                impl->fp8fp8fp32.heuristicResult, &impl->fp8fp8fp32.algoNum));
        if (impl->fp8fp8fp32.algoNum == 0) {
            CHECK_CUBLAS(CUBLAS_STATUS_NOT_SUPPORTED);
        }
        *instance = impl;
        return CUBLAS_STATUS_SUCCESS;
    }

    cublasStatus_t cublasLtFp8RowMajorNTNMeta::matmul(
            void *instance, cudaStream_t stream, double alpha, const void *A,
            const void *B, double beta, void *C, cudaDataType Ctype, int algoIdx) {
        auto impl = static_cast<Impl *>(instance);
        float alphaFP32 = alpha;
        float betaFP32 = beta;
        switch (Ctype) {
            case CUDA_R_32F:
                return cublasLtMatmul(impl->handle, impl->operationDesc, &alphaFP32, B,
                                      impl->Bdesc, A, impl->Adesc, &betaFP32, C,
                                      impl->fp8fp8fp32.Cdesc, C, impl->fp8fp8fp32.Ddesc,
                                      &impl->fp8fp8fp32.heuristicResult[algoIdx].algo,
                                      impl->workspace, impl->workspaceSize, stream);
            case CUDA_R_16F:
                return cublasLtMatmul(impl->handle, impl->operationDesc, &alphaFP32, B,
                                      impl->Bdesc, A, impl->Adesc, &betaFP32, C,
                                      impl->fp8fp8fp16.Cdesc, C, impl->fp8fp8fp16.Ddesc,
                                      &impl->fp8fp8fp16.heuristicResult[algoIdx].algo,
                                      impl->workspace, impl->workspaceSize, stream);
            default:
                return CUBLAS_STATUS_NOT_SUPPORTED;
        }
    }

    cublasStatus_t cublasLtFp8RowMajorNTNMeta::getAlgoNum(
            void *instance, cudaStream_t stream, double alpha, const void *A,
            const void *B, double beta, void *C, cudaDataType Ctype, int *algoNum) {
        auto impl = static_cast<Impl *>(instance);
        float alphaFP32 = alpha;
        float betaFP32 = beta;
        switch (Ctype) {
            case CUDA_R_32F:
                *algoNum = impl->fp8fp8fp32.algoNum;
                return CUBLAS_STATUS_SUCCESS;
            case CUDA_R_16F:
                *algoNum = impl->fp8fp8fp16.algoNum;
                return CUBLAS_STATUS_SUCCESS;
            default:
                return CUBLAS_STATUS_NOT_SUPPORTED;
        }
    }

    cublasStatus_t cublasLtFp8RowMajorNTNMeta::destroy(void *instance) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties_v2(&deviceProp, 0);
        if (deviceProp.major < 9) {
            return CUBLAS_STATUS_SUCCESS;
        }
        auto impl = static_cast<Impl *>(instance);
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(impl->fp8fp8fp32.Ddesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(impl->fp8fp8fp32.Cdesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(impl->fp8fp8fp16.Ddesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(impl->fp8fp8fp16.Cdesc));
        CHECK_CUBLAS(cublasLtMatmulDescDestroy(impl->operationDesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(impl->Adesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(impl->Bdesc));
        CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(impl->preference));
        CHECK_CUBLAS(cublasLtDestroy(impl->handle));
        return CUBLAS_STATUS_SUCCESS;
    }
}  // namespace mixed_kernels

int main() {
    void *instance;
    const uint64_t m = 4096;
    const uint64_t n = 4096;
    const uint64_t k = 4096;
    const uint64_t lda = k;
    const uint64_t ldb = k;
    const uint64_t ldc = n;
    void *aPrt;
    void *bPrt;
    void *cPrt;
    std::size_t workspaceSize = 40960;
    void *workspace;
    cudaStream_t stream;
    cudaEvent_t start, end;
    const int repeat = 3;
    const uint64_t flopsPerMatrixMul = 2 * m * n * k;
    const uint64_t totalFlops = flopsPerMatrixMul * repeat;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "CUDA Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    if (deviceProp.major < 9) {
        printf("Your device does not support FP8");
        return 1;
    }

    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));
    CHECK_CUDA(cudaMalloc(&aPrt, m * k * 1));
    CHECK_CUDA(cudaMemset(aPrt, 0, m * k * 1));
    CHECK_CUDA(cudaMalloc(&bPrt, n * k * 1));
    CHECK_CUDA(cudaMemset(bPrt, 0, n * k * 1));
    CHECK_CUDA(cudaMalloc(&cPrt, m * n * sizeof(float)));
    CHECK_CUDA(cudaMemset(cPrt, 0, m * n * sizeof(float)));
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));

    mixed_kernels::cublasLtFp8RowMajorNTNMeta::create(&instance, m, n, k, lda, ldb, ldc, workspaceSize, workspace);

    // warm up
    for (int i = 0; i < 3; ++i) {
        mixed_kernels::cublasLtFp8RowMajorNTNMeta::matmul(instance, stream, 1., aPrt, bPrt, 1., cPrt, CUDA_R_32F);
    }

    // warm up
    for (int i = 0; i < 3; ++i) {
        mixed_kernels::cublasLtFp8RowMajorNTNMeta::matmul(instance, stream, 1., aPrt, bPrt, 1., cPrt, CUDA_R_16F);
    }

    int algoNumFp32;
    int algoNumFp16;
    mixed_kernels::cublasLtFp8RowMajorNTNMeta::getAlgoNum(instance, stream, 1., aPrt, bPrt, 1., cPrt, CUDA_R_32F,
                                                          &algoNumFp32);
    mixed_kernels::cublasLtFp8RowMajorNTNMeta::getAlgoNum(instance, stream, 1., aPrt, bPrt, 1., cPrt, CUDA_R_16F,
                                                          &algoNumFp16);

    for (int i = 0; i < algoNumFp32; ++i) {
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaEventRecord(start, stream));
        for (int j = 0; j < repeat; ++j) {
            mixed_kernels::cublasLtFp8RowMajorNTNMeta::matmul(instance, stream, 1., aPrt, bPrt, 1., cPrt, CUDA_R_32F);
        }
        CHECK_CUDA(cudaEventRecord(end, stream));
        CHECK_CUDA(cudaEventSynchronize(end));

        // Calculate and print the elapsed time
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, end));
        std::cout << "---------------------------------" << std::endl;
        std::cout << "FP32 Algo " << i << ": Elapsed time: " << milliseconds << " ms\n";
        const double elapsedSeconds = milliseconds / 1000.0f;
        const double Gflops = totalFlops / elapsedSeconds / 1e9;
        std::cout << "GFLOPS: " << Gflops << std::endl;
    }

    for (int i = 0; i < algoNumFp16; ++i) {
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaEventRecord(start, stream));
        for (int j = 0; j < repeat; ++j) {
            mixed_kernels::cublasLtFp8RowMajorNTNMeta::matmul(instance, stream, 1., aPrt, bPrt, 1., cPrt, CUDA_R_16F);
        }
        CHECK_CUDA(cudaEventRecord(end, stream));
        CHECK_CUDA(cudaEventSynchronize(end));

        // Calculate and print the elapsed time
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, end));
        std::cout << "---------------------------------" << std::endl;
        std::cout << "FP16 Algo " << i << ": Elapsed time: " << milliseconds << " ms\n";
        const double elapsedSeconds = milliseconds / 1000.0f;
        const double Gflops = totalFlops / elapsedSeconds / 1e9;
        std::cout << "GFLOPS: " << Gflops << std::endl;
    }
    std::cout << "---------------------------------" << std::endl;

    CHECK_CUDA(cudaEventDestroy(end));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(cPrt));
    CHECK_CUDA(cudaFree(bPrt));
    CHECK_CUDA(cudaFree(aPrt));
    CHECK_CUDA(cudaFree(workspace));
    mixed_kernels::cublasLtFp8RowMajorNTNMeta::destroy(instance);
}