
#include <cstdlib>
#include <chrono>
#include <iostream>   
#include <iomanip>
#include "vector.h"
#include <cmath>

//References:
//https://devblogs.nvidia.com/using-shared-memory-cuda-cc/
//https://docs.nvidia.com/cuda/cublas/index.html
//https://docs.nvidia.com/cuda/cusolver/index.html

#include <cstdio>
#include <cuda_runtime.h>
#include <assert.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) 
            exit(code);
    }
}

namespace vector {


//Return: ae^(kt) + be^(qt) + c
__global__ void DoubleExp(float* A, float a, float b, float c, float k, float q, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        A[i] = a * expf(k * i) + b * expf(q * i) + c;
}
__host__ float* DoubleExp(float* param, int N){
    float* A;
    gpuErrchk(cudaMalloc(&A, N*sizeof(float)));
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    DoubleExp<<<blocksPerGrid, threadsPerBlock>>>(A, param[0], param[1], param[2], param[3], param[4], N);
    return A;
}

//Copy vector. A = B; 
__global__ void CopyVecf(float* A, float* B, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        A[i] = B[i];
}
__host__ void CopyVecfh(float* A, float* B, int N){
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    CopyVecf<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
}

//Calculates the Jacobian of f(t) for K parameters and N data points
//Returns N x K matrix in column major order

__host__ float* Jacobian(float* f(float* param, int N), float* param, int N, int K){
    const float eps = pow(2,-17);
    float* J;
    gpuErrchk(cudaMalloc(&J, N * K * sizeof(float)));

    float* prev = f(param, N);
    float* next;

    float* debug = (float*)malloc(N * K * sizeof(float));


    for (int i = 0; i < K; i++){
        param[i] += eps;
        next = f(param, N);
        SubVecfh(next, prev, (float*)&J[i * N], N);
        param[i] -= eps;
        cudaFree(next);
    }
    
    cudaFree(prev);
    return ScaleVecf(J, float(1./eps), N * K);
}

//Same as Jacboian() but uses adaptive values of epsilon scaled with the value of param.
//This can achieve better precisions across a wider range of values.
//TODO: Consider logarithms then addition instead of multiplications. Reduces arithmetic error.
__host__ float* Jacobian2(float* f(float* param, int N), float* param, int N, int K){
    const float eps = pow(2,-10);
    float* J;
    gpuErrchk(cudaMalloc(&J, N * K * sizeof(float)));

    float* prev = f(param, N);
    float* next;
    float temp, dB, scale;

    for (int i = 0; i < K; i++){
        temp = param[i];
        dB = param[i] * eps;
        if (dB < 1e-31 && dB > -1e-31)
            dB = 1e-30;
        //dB^-1. Recalculation for precision. 
        scale = (float) (1./((double)param[i] * eps));
        param[i] += dB;
        next = f(param, N);
        SubVecfh(next, prev, (float*)&J[i * N], N);
        ScaleVecfh((float*)&J[i * N], scale, (float*)&J[i * N], N);
        param[i] = temp;
        cudaFree(next);
    }
    
    cudaFree(prev);
    return J;
}
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
    cudaMemset(C, 0, M * N * sizeof(float));

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

__host__ void SVD(float* A, int M, int N, float* S, float* U, float* VT){ 
    
    // float* debug = (float*)malloc(M * N * sizeof(float));
    // cudaMemcpy(debug, A, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < M; i++){
    //     for (int j = 0; j < N; j++){
    //         std::cout << std::left << std::setw(15) << debug[i + j * M];
    //     }
    //     std::cout << '\n';
    // }
    
    cusolverDnHandle_t handle;
    cusolverStatus_t stat = cusolverDnCreate(&handle);
    if (stat != CUSOLVER_STATUS_SUCCESS) {
        printf ("CUSOLVER initialization failed\n");
        exit(EXIT_FAILURE);
    }

    int* devInfo; gpuErrchk(cudaMalloc(&devInfo, sizeof(int))); 
    int lwork;

    stat = cusolverDnSgesvd_bufferSize(
        handle,
        M,
        N,
        &lwork);
    if (stat != CUSOLVER_STATUS_SUCCESS) {
        printf ("SVD buffer failed\n");
        exit(EXIT_FAILURE);
    }

    float* work; gpuErrchk(cudaMalloc(&work, lwork * sizeof(float)));

    stat = cusolverDnSgesvd(
        handle,
        'A',
        'A',
        M,
        N,
        A,
        M,
        S,
        U,
        M,
        VT,
        N,
        work,
        lwork,
        NULL,
        devInfo);

    if (stat != CUSOLVER_STATUS_SUCCESS) {
        printf ("SVD Solve failed\n");
        int dev = 0;
        cudaMemcpy(&dev, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        printf ("Code: %d, Devinfo: %d \n", (int)stat, dev);
        printf ("M:%d N:%d lda:%d ldu:%d ldvt:%d", M, N, M, M, N);
        exit(EXIT_FAILURE);
    }

    cudaFree(work);
    cudaFree(devInfo);
}

__host__ void MPInverse(float* A, int M, int N){

    //Cusolver SVD requires M >= N

    float* S; gpuErrchk(cudaMalloc(&S, N * N * sizeof(float)));
    float* U; gpuErrchk(cudaMalloc(&U, M * M * sizeof(float)));
    float* VT; gpuErrchk(cudaMalloc(&VT, N * N * sizeof(float)));

    SVD(A, M, N, S, U, VT);
    
}

//Gauss-Newton method
__host__ void FitGaussNewton(float* A, float* param, float* f(float* param, int N), int N, int K){

    //Finding inverse Jacobian
    float* J = Jacobian(f, param, N, K);
    float* S, *U, *VT;
    int P = N;
    if (N > K)
        P = K;

    gpuErrchk(cudaMalloc(&S, P * P * sizeof(float)));
    gpuErrchk(cudaMalloc(&U, N * N * sizeof(float)));
    gpuErrchk(cudaMalloc(&VT, K * K * sizeof(float)));
    SVD(J, N, K, S, U, VT);


    // cublasSgeam(handle,
    //     CUBLAS_OP_T, CUBLAS_OP_N,
    //     int m, int n,
    //     const float           *alpha,
    //     const float           *A, int lda,
    //     const float           *beta,
    //     const float           *B, int ldb,
    //     float           *C, int ldc);

    cudaFree(S);
    cudaFree(U);
    cudaFree(VT);
}

}

__host__ void Print(const float* A, int M, int N){
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            std::cout << std::left << std::setw(15) << A[i + j * M];
        }
        std::cout << '\n';
    }
}

void TestMult(){
    int M = 2;
    int N = 4;
    int K = 3;
    float A[8] = {3, 9, 2, 1, 1, 3, 5, 0};
    float B[12] = {2, 1, 2, 8, 9, 3, 4 ,1, 0, 5, 7, 5};
    float* C = (float*)malloc(M * K * sizeof(float));
    float* A_d; gpuErrchk(cudaMalloc(&A_d, M * N * sizeof(float)));
    float* B_d; gpuErrchk(cudaMalloc(&B_d, N * K * sizeof(float)));
    float* C_d; gpuErrchk(cudaMalloc(&C_d, M * K * sizeof(float)));
    cudaMemcpy(A_d, &A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, &B, N * K * sizeof(float), cudaMemcpyHostToDevice);
    vector::MultMat(A_d, B_d, C_d, 1, 1, M, N, K);
    
    cudaMemcpy(C, C_d, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    

    Print(C, M, K);
}

void TestSVD(){
    int M = 3;
    int N = 2;

    int P = M;
    if (M > N)
        P = N;

    float* Sh =     (float*)malloc(P * sizeof(float));
    float* Uh =     (float*)malloc(M * M * sizeof(float));
    float* VTh =    (float*)malloc(N * N * sizeof(float));
    float* result_h=(float*)malloc(M * N * sizeof(float)); 
    float* S;       gpuErrchk(cudaMalloc(&S, P * sizeof(float)));
    float* U;       gpuErrchk(cudaMalloc(&U, M * M * sizeof(float)));
    float* VT;      gpuErrchk(cudaMalloc(&VT, N * N * sizeof(float)));
    float* W;       gpuErrchk(cudaMalloc(&W, M * N * sizeof(float)));

    float A_h[20] = {
        2,0,0,
        0,-3,0};
    float* A_d;
    gpuErrchk(cudaMalloc(&A_d, M * N * sizeof(float)));
    cudaMemcpy(A_d, &A_h, M * N * sizeof(float), cudaMemcpyHostToDevice);

    vector::SVD(A_d, M, N, S, U, VT);
    
    cudaMemcpy(Sh, S, P * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Uh, U, M * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(VTh, VT, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "\nU:\n";    Print(Uh, M, M);
    std::cout << "\nS:\n";    Print(Sh, 1, N);
    std::cout << "\nVT:\n";   Print(VTh, N, N);
    std::cout << "\n\n";

    vector::MultMatD(S, VT, W, N, N);
    vector::MultMat(U, W, A_d, 1, 1, M, N, N);
    
    cudaMemcpy(result_h, A_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    Print(result_h, M, N);
}

int main(){
    int N = 1000;
    size_t size = N * sizeof(float);
    float T = 1e-9;
    float* voltage = (float*)malloc(size);
    float* d_voltage;
    gpuErrchk(cudaMalloc(&d_voltage, size));

    // voltage[0] = 0;
    // for (int i = 1; i < N; i++){
    //     voltage[i] = voltage[i-1] * 0.99998000019999866667;
    //     if(rand()%10000 == 0)
    //         voltage[i] += 1;
    // }

    for (int i = 0; i < N; i++){
        voltage[i] = i;
    }

    voltage[0] = 100;
    for (int i = 1; i < N; i++){
        voltage[i] = voltage[i-1] * 0.99998000019999866667;
    }

    //Moving data to GPU
    cudaMemcpy(d_voltage, voltage, size, cudaMemcpyHostToDevice);

    //Fitting V = ae^(kt) + be^(qt) + c
    //param: {a, b, c, k, q}
    float param[5] = {
        1,
        -1,
        1,
        -1 * 1e-3,
        -10 * 1e-3
    };



    TestSVD();



    cudaFree(d_voltage);
    cudaFree(d_result);
    free(result);
    free(voltage);
    
}
