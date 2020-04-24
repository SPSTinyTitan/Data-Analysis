
#include <cstdlib>
#include <iostream>   
#include <iomanip>
#include <cmath>

#include "vector_wrappers.h"
#include "matrix_wrappers.h"
#include "cuda_tools.h"
#include "fitting.h"


//References:
//https://devblogs.nvidia.com/using-shared-memory-cuda-cc/
//https://docs.nvidia.com/cuda/cublas/index.html
//https://docs.nvidia.com/cuda/cusolver/index.html

namespace fit{

    // Applies lambda function to a vector
    template<typename f>
    __device__ float apply__(float A, float B, f lambda){
        return lambda(A, B);
    }
    template<typename f>
    __global__ void apply_(float* A, float B, float* C, f lambda, int N){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < N)
            C[i] = apply__(A[i], B, lambda);
    }
    template<typename f>
    __host__ void apply(float* A, float B, float* C, f lambda, int N){
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        apply_<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, lambda, N);
        gpuErrchk(cudaDeviceSynchronize());
    }
    __host__ void invertS(float* S, float* invS, int N){
        auto lambda = []__device__(float a, float b){return (a > b) ? (1./a) : 0;};
        float eps = 1e-25 * vector::dot(S, S, N);
        std::cout << eps << ":\n";
        //print_(S, 1, N);
        apply(S, eps, S, lambda, N);
        //print_(S, 1, N);
    }

    __host__ void pInv(float* A, int M, int N){

        int P = M;
        if (M > N) P = N;

        float* S;       gpuErrchk(cudaMalloc(&S, M * sizeof(float)));
        float* U;       gpuErrchk(cudaMalloc(&U, M * M * sizeof(float)));
        float* VT;      gpuErrchk(cudaMalloc(&VT, N * N * sizeof(float)));
        float* W;       gpuErrchk(cudaMalloc(&W, M * N * sizeof(float)));
        gpuErrchk(cudaMemset(W, 0, M * N * sizeof(float)));
        gpuErrchk(cudaMemset(S, 0, M * sizeof(float)));
        svd(A, M, N, S, U, VT);

        matrix::multD(S, VT, W, N, M);
        matrix::mult(U, W, A, 1, 1, false, false, M, N, N);
        invertS(S, S, P);


        matrix::multD(S, VT, W, N, M);
        matrix::mult(U, W, A, 1, 1, false, false, M, N, N);
        matrix::transpose(A, A, M, N);

        if (S)  cudaFree(S);
        if (U)  cudaFree(U);
        if (VT) cudaFree(VT);
        if (W)  cudaFree(W);

    }

    //Gauss-Newton method
    __host__ void gaussNewton(float* A, float* param, void f(float* A, float* param, int N), int N, int K){

        float eps = 0.05;

        //Finding inverse of Jacobian
        float* J;       gpuErrchk(cudaMalloc(&J, N * K * sizeof(float)));
        float* F;       gpuErrchk(cudaMalloc(&F, N * sizeof(float)));
        float* P;       gpuErrchk(cudaMalloc(&P, K * sizeof(float)));
        float* P_h =    (float*)malloc(K *sizeof(float));
        float error;
        int count = 0;
        do{
            count++;
            //J^-1
            jacobian_v2(J, f, param, N, K);
            pInv(J, N, K);
            
            //(y - f(param))
            doubleExp(F, param, N);
            vector::sub(A, F, F, N);
            
            //J^-1(y - f(param))
            matrix::mult(J, F, P, 1, 1, false, false, K, N, 1);

            //Update param
            cudaMemcpy(P_h, P, K * sizeof(float), cudaMemcpyDeviceToHost);
            for (int i = 0; i < K; i++)
                param[i] += eps * P_h[i];
            
            //Print errors
            // doubleExp(F, param, N);
            // vector::sub(A, F, F, N);
            // std::cout << "Error:" << vector::sum(F, N) << '\n';

            error = vector::sum(F, N)/N;

        }while(error > 0.01);
        
        std::cout << "Converged in " << count << "iterations\n";
        if (J)  cudaFree(J);
        if (F)  cudaFree(F);
        if (P)  cudaFree(P);
        if (P_h) free(P_h);
    }

}

void TestMult(){
    int M = 4;
    int N = 4;
    int K = 3;
    float A[16] = {3, 9, 2, 1, 1, 3, 5, 0, 3, 9, 2, 1, 1, 3, 5, 0};
    float B[12] = {2, 1, 2, 8, 9, 3, 4 ,1, 0, 5, 7, 5};
    float* A_d; gpuErrchk(cudaMalloc(&A_d, M * N * sizeof(float)));
    float* B_d; gpuErrchk(cudaMalloc(&B_d, N * K * sizeof(float)));
    float* C_d; gpuErrchk(cudaMalloc(&C_d, M * K * sizeof(float)));
    cudaMemcpy(A_d, &A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, &B, N * K * sizeof(float), cudaMemcpyHostToDevice);
    matrix::mult(A_d, B_d, C_d, 1, 1, false, false, M, N, K);
    print_(C_d, M, K);
    std::cout << "\nTranspose: \n";
    matrix::transpose(C_d, C_d, M, K);
    print_(C_d, K, M);

    //Expected Output:

    // 21             43             31             
    // 63             129            93             
    // 53             46             64             
    // 4              13             7              
    
    // Transpose: 
    // 21             63             53             4              
    // 43             129            46             13             
    // 31             93             64             7   
}

void TestSvd(){
    int M = 3;
    int N = 2;

    int P = M;
    if (M > N)
        P = N;

    float* result_h = (float*)malloc(M * N * sizeof(float)); 
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

    fit::svd(A_d, M, N, S, U, VT);
    
    std::cout << "\nU:\n";    print_(U, M, M);
    std::cout << "\nS:\n";    print_(S, 1, N);
    std::cout << "\nVT:\n";   print_(VT, N, N);
    std::cout << "\n\n";

    matrix::multD(S, VT, W, N, M);
    matrix::mult(U, W, A_d, 1, 1, false, false, M, N, N);
    
    print_(A_d, M, N);
}

int main(){
    int N = 5000;
    size_t size = N * sizeof(float);
    float T = 1e-9;
    float* voltage = (float*)malloc(size);
    float* d_voltage;   gpuErrchk(cudaMalloc(&d_voltage, size));
    float* d_fit;       gpuErrchk(cudaMalloc(&d_fit, size));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

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
        voltage[i] = voltage[i-1] * 0.99998;
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

    cudaEventRecord(start);
    fit::gaussNewton(d_voltage, param, fit::doubleExp, N, 5);
    cudaEventRecord(stop);
    fit::doubleExp(d_fit, param, N);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << milliseconds << "\n\n";
    //print_(d_fit, N, 1);
    //print_(d_voltage, N, 1);
    print(param, 1, 5);

    //TestMult();
    //TestSvd();
    cudaFree(d_voltage);
    free(voltage);
    
}
