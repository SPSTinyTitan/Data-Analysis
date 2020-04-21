
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
#include <cusolverDn.h>
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
        if (dB < 1e-35 && dB > -1e-35)
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

__host__ void SVD(float* A, int M, int N, float* S, float* U, float* VT){ 
    
    float* debug = (float*)malloc(M * N * sizeof(float));
    cudaMemcpy(debug, A, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            std::cout << std::left << std::setw(15) << debug[i + j * M];
        }
        std::cout << '\n';
    }
    
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
        'N',
        'N',
        M,
        N,
        A,
        M,
        S,
        NULL,
        M,
        NULL,
        N,
        work,
        lwork,
        NULL,
        devInfo);
    
    if (stat != CUSOLVER_STATUS_SUCCESS) {
        printf ("S Solve failed\n");
        int* dev = (int*)malloc(sizeof(int));
        cudaMemcpy(dev, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        printf ("Code: %d, Devinfo: %d \n", (int)stat, *dev);
        printf ("M:%d N:%d lda:%d ldu:%d ldvt:%d", M, N, M, M, N);
        exit(EXIT_FAILURE);
    }

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
        int* dev = (int*)malloc(sizeof(int));
        cudaMemcpy(dev, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
        printf ("Code: %d, Devinfo: %d \n", (int)stat, *dev);
        printf ("M:%d N:%d lda:%d ldu:%d ldvt:%d", M, N, M, M, N);
        exit(EXIT_FAILURE);
    }


    cudaFree(work);
    cudaFree(devInfo);
    printf ("%d", (int)CUSOLVER_STATUS_INVALID_VALUE);

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
    float* d_result = vector::Jacobian2(vector::DoubleExp, param, N, 5);
    //float* d_result = vector::DoubleExp(param, N);
    cudaMemcpy(result, d_result, size * 5, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < N; i++){
    //     std::cout << std::left 
    //         << std::setw(15) << i
    //         << std::setw(15) << result[i]
    //         << std::setw(15) << result[i+N]
    //         << std::setw(15) << result[i+2*N]
    //         << std::setw(15) << result[i+3*N]
    //         << std::setw(15) << result[i+4*N]
    //         << '\n';
    // }

    int M = 3;
    N = 5;
    //float* J = d_result;
    //float A[15] = {1,1,1,0,0,1,0,0,1,1,1,0,1,0,1};
    float A[15] = {1,1,1,1,0,0,1,0,1,0,1,0,0,1,1};
    float* J;
    gpuErrchk(cudaMalloc(&J, M * N * sizeof(float)));
    cudaMemcpy(J, &A, M * N * sizeof(float), cudaMemcpyHostToDevice);

    float* S, *U, *VT;
    int P = M;
    if (M > N)
        P = N;


    float* Sh = (float*)malloc(P * sizeof(float));
    float* Uh = (float*)malloc(M * M * sizeof(float));
    float* VTh = (float*)malloc(N * N * sizeof(float));
    gpuErrchk(cudaMalloc(&S, P * P * sizeof(float)));
    gpuErrchk(cudaMalloc(&U, M * M * sizeof(float)));
    gpuErrchk(cudaMalloc(&VT, N * N * sizeof(float)));

    vector::SVD(J, M, N, S, U, VT);
    
    cudaMemcpy(Sh, S, P * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Uh, U, M * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(VTh, VT, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++){
        for (int j = 0; j < M; j++){
            std::cout << std::left << std::setw(15) << Uh[i + j * M];
        }
        std::cout << '\n';
    }
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            std::cout << std::left << std::setw(15) << VTh[i + j * N];
        }
        std::cout << '\n';
    }

    // int K = 5;
    // float* J = d_result;
    // float* S, *U, *VT;
    // int P = N;
    // if (N > K)
    //     P = K;

    // gpuErrchk(cudaMalloc(&S, P * P * sizeof(float)));
    // gpuErrchk(cudaMalloc(&U, N * N * sizeof(float)));
    // gpuErrchk(cudaMalloc(&VT, K * K * sizeof(float)));
    // vector::SVD(J, N, K, S, U, VT);

    // float* Sh = (float*)malloc(size);
    // float* Uh = (float*)malloc(size * K);
    // float* VTh = (float*)malloc(size * K);
    // cudaMemcpy(Sh, S, size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(Uh, U, size * K, cudaMemcpyDeviceToHost);
    // cudaMemcpy(VTh, VT, size * K, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < N; i++){
    //     std::cout << std::left 
    //         << std::setw(15) << Sh[i]
    //         << std::setw(15) << Uh[i]
    //         << std::setw(15) << VTh[i]
    //         << '\n';
    // }

    cudaFree(d_voltage);
    cudaFree(d_result);
    free(result);
    free(voltage);
    
}