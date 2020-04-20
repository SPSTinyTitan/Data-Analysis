
#include <cstdlib>
#include <chrono>
#include <iostream>   
#include <iomanip>
#include "vector.h"

//References:
//https://devblogs.nvidia.com/using-shared-memory-cuda-cc/
//https://docs.nvidia.com/cuda/cublas/index.html
//https://docs.nvidia.com/cuda/cusolver/index.html

#include <cstdio>
#include "cublas_v2.h"
#include <cuda_runtime.h>
#include "cusolverDn.h"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
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
const float eps = 1e-6;
__host__ float* Jacobian(float* f(float* param, int N), float* param, int N, int K){
    float* J;
    gpuErrchk(cudaMalloc(&J, N * K * sizeof(float)));

    float* prev = f(param, N);
    float* next;

    float* debug = (float*)malloc(N * sizeof(float));


    for (int i = 0; i < K; i++){
        param[i] += eps;
        next = f(param, N);
        
        cudaMemcpy(debug, next, N*sizeof(float),cudaMemcpyDeviceToHost);
        for (int j = 0; j < N; j++){
            std::cout << debug[j] << '\n';
        }

        SubVecfh(next, prev, (float*)&J[i * N], N);
        param[i] -= eps;
        cudaFree(next);
    }
    cudaFree(prev);
    return ScaleVecf(J, float(1./eps), N * K);
}

__host__ void SVD(float* A, int M, int N, float* S, float* U, float* VT){  
    cusolverDnHandle_t handle;
    cusolverStatus_t stat = cusolverDnCreate(&handle);
    if (stat != CUSOLVER_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        exit(EXIT_FAILURE);
    }

    float* work;
    int lwork;
    int* devInfo;

    cusolverDnSgesvd_bufferSize(
        handle,
        M,
        N,
        &lwork);

    gpuErrchk(cudaMalloc(&work, lwork * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&devInfo,sizeof(int))); 

    cusolverDnSgesvd(
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
    
    cudaFree(work);
    cudaFree(devInfo);
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
        -10 * 1e-3};

    float* result = (float*) malloc(size * 5);
    float* d_result = vector::Jacobian(vector::DoubleExp, param, N, 5);
    //float* d_result = vector::DoubleExp(param, N);
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < N; i++){
    //     std::cout << std::left 
    //         << std::setw(15) << result[i]
    //         << std::setw(15) << result[i+N]
    //         << std::setw(15) << result[i+2*N]
    //         << std::setw(15) << result[i+3*N]
    //         << std::setw(15) << result[i+4*N]
    //         << '\n';
    // }

    cudaFree(d_voltage);
    cudaFree(d_result);
    free(result);
    free(voltage);
    
}