#include <matrix_wrappers.h>

namespace matrix{

    //C = (alpha * A) x (beta * B)
    //A - M x N
    //B - N x K
    //C - M x K
    __host__ void MultMat(float* A, float* B, float* C, float alpha, float beta, int M, int N, int K){

        //Initializing CuBlas
        cublasHandle_t cublasH = NULL;
        cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
        cublas_status = cublasCreate(&cublasH);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);

        //Zeroing results
        cudaMemset(C, 0, M * K * sizeof(float));

        //Multiplying
        cublas_status = cublasSgemm_v2(
            cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, K, N,
            &alpha,
            A, M,
            B, N,
            &beta,
            C, M
        );
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    }

    //Calculates diagonal matrix multiplication
    //C = A x B
    //A - 1 x N diagonal (representing MxN)
    //B - N x K
    __host__ void MultMatD(float* A, float* B, float* C, int M, int N){
        
        //Initializing CuBlas
        cublasHandle_t cublasH = NULL;
        cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
        cublas_status = cublasCreate(&cublasH);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);

        //Zeroing results
        gpuErrchk(cudaMemset(C, 0, M * N * sizeof(float)));

        //Multiplying
        cublas_status = cublasSdgmm(
            cublasH, CUBLAS_SIDE_LEFT,
            M, N,
            B, M,
            A, 1,
            C, M
        );
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    }

}