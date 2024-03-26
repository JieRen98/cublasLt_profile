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
                    cublasLtMatmulHeuristicResult_t heuristicResult{};
                } fp8fp8fp16{}, fp8fp8fp32{};
            };
        }  // namespace

        CUBLASAPI cublasStatus_t CUBLASWINAPI
        create(void **instance, uint64_t m, uint64_t n, uint64_t k, int64_t lda,
               int64_t ldb, int64_t ldc, std::size_t workspaceSize, void *workspace);

        CUBLASAPI cublasStatus_t CUBLASWINAPI destroy(void *instance);

        static CUBLASAPI cublasStatus_t CUBLASWINAPI
        matmul(void *instance, cudaStream_t stream, double alpha, const void *A,
               const void *B, double beta, void *C, cudaDataType Ctype);
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

        int returnedResults;
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&impl->fp8fp8fp16.Cdesc, CUDA_R_16F,
                                                n, m, ldc));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&impl->fp8fp8fp16.Ddesc, CUDA_R_16F,
                                                n, m, ldc));
        CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
                impl->handle, impl->operationDesc, impl->Bdesc, impl->Adesc,
                impl->fp8fp8fp16.Cdesc, impl->fp8fp8fp16.Ddesc, impl->preference, 1,
                &impl->fp8fp8fp16.heuristicResult, &returnedResults));
        if (returnedResults == 0) {
            CHECK_CUBLAS(CUBLAS_STATUS_NOT_SUPPORTED);
        }

        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&impl->fp8fp8fp32.Cdesc, CUDA_R_32F,
                                                n, m, ldc));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&impl->fp8fp8fp32.Ddesc, CUDA_R_32F,
                                                n, m, ldc));
        CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
                impl->handle, impl->operationDesc, impl->Bdesc, impl->Adesc,
                impl->fp8fp8fp32.Cdesc, impl->fp8fp8fp32.Ddesc, impl->preference, 1,
                &impl->fp8fp8fp32.heuristicResult, &returnedResults));
        if (returnedResults == 0) {
            CHECK_CUBLAS(CUBLAS_STATUS_NOT_SUPPORTED);
        }
        *instance = impl;
        return CUBLAS_STATUS_SUCCESS;
    }

    cublasStatus_t cublasLtFp8RowMajorNTNMeta::matmul(
            void *instance, cudaStream_t stream, double alpha, const void *A,
            const void *B, double beta, void *C, cudaDataType Ctype) {
        auto impl = static_cast<Impl *>(instance);
        float alphaFP32 = alpha;
        float betaFP32 = beta;
        switch (Ctype) {
            case CUDA_R_32F:
                return cublasLtMatmul(impl->handle, impl->operationDesc, &alphaFP32, B,
                                      impl->Bdesc, A, impl->Adesc, &betaFP32, C,
                                      impl->fp8fp8fp32.Cdesc, C, impl->fp8fp8fp32.Ddesc,
                                      &impl->fp8fp8fp32.heuristicResult.algo,
                                      impl->workspace, impl->workspaceSize, stream);
            case CUDA_R_16F:
                return cublasLtMatmul(impl->handle, impl->operationDesc, &alphaFP32, B,
                                      impl->Bdesc, A, impl->Adesc, &betaFP32, C,
                                      impl->fp8fp8fp16.Cdesc, C, impl->fp8fp8fp16.Ddesc,
                                      &impl->fp8fp8fp16.heuristicResult.algo,
                                      impl->workspace, impl->workspaceSize, stream);
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

    for (int i = 0; i < 3; ++i) {
        mixed_kernels::cublasLtFp8RowMajorNTNMeta::matmul(instance, stream, 1., aPrt, bPrt, 1., cPrt, CUDA_R_32F);
    }

    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaEventRecord(start, stream));

    // Perform the operation you want to time
    for (int i = 0; i < 3; ++i) {
        mixed_kernels::cublasLtFp8RowMajorNTNMeta::matmul(instance, stream, 1., aPrt, bPrt, 1., cPrt, CUDA_R_32F);
    }

    // Stop recording
    CHECK_CUDA(cudaEventRecord(end, stream));
    CHECK_CUDA(cudaEventSynchronize(end));

    // Calculate and print the elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, end));
    std::cout << "Elapsed time: " << milliseconds << " ms\n";

    CHECK_CUDA(cudaEventDestroy(end));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(cPrt));
    CHECK_CUDA(cudaFree(bPrt));
    CHECK_CUDA(cudaFree(aPrt));
    CHECK_CUDA(cudaFree(workspace));
    mixed_kernels::cublasLtFp8RowMajorNTNMeta::destroy(instance);
}