
#include <cstdlib>
#include <iostream>   
#include <iomanip>
#include <cmath>

#include "vector_wrappers.h"
#include "matrix_wrappers.h"
#include "fitting.h"
#include "cuda_tools.h"



//References:
//https://devblogs.nvidia.com/using-shared-memory-cuda-cc/
//https://docs.nvidia.com/cuda/cublas/index.html
//https://docs.nvidia.com/cuda/cusolver/index.html


namespace fit{

    __host__ void pInv(float* A, int M, int N){

        int P = M;
        if (M > N) P = N;

        float* S;       gpuErrchk(cudaMalloc(&S, P * P * sizeof(float)));
        float* U;       gpuErrchk(cudaMalloc(&U, M * M * sizeof(float)));
        float* VT;      gpuErrchk(cudaMalloc(&VT, N * N * sizeof(float)));
        float* W;       gpuErrchk(cudaMalloc(&W, M * N * sizeof(float)));
        
        svd(A, M, N, S, U, VT);

        //Check results for correctness

        // std::cout << "S:\n" << std::endl;
        // print_(S, 1, P);
        // std::cout << "\n\nU:\n" << std::endl;
        // print_(U, M, M);
        // std::cout << "\n\nVT:\n" << std::endl;
        // print_(VT, N, N);
        
        // matrix::multD(S, VT, W, N, N);
        // matrix::mult(U, W, J, 1, 1, false, false, M, N, N);

        // print_(J, M, N);
        vector::div(1,S,S,P);
        


        print_(S, 1, P);

        if (S)  cudaFree(S);
        if (U)  cudaFree(U);
        if (VT) cudaFree(VT);
        if (W)  cudaFree(W);

    }

    //Gauss-Newton method
    __host__ void FitGaussNewton(float* A, float* param, void f(float* A, float* param, int N), int N, int K){

        int P = N;
        if (N > K) P = K;

        //Finding inverse of Jacobian
        float* J;       gpuErrchk(cudaMalloc(&J, N * K * sizeof(float)));
        
        jacobian_v2(J, f, param, N, K);
        pInv(J, N, K);
        

        if (J)  cudaFree(J);

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
    float* D_d; gpuErrchk(cudaMalloc(&D_d, M * K * sizeof(float)));
    cudaMemcpy(A_d, &A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, &B, N * K * sizeof(float), cudaMemcpyHostToDevice);
    matrix::mult(A_d, B_d, C_d, 1, 1, false, false, M, N, K);
    print_(C_d, M, K);
    matrix::transpose(C_d, D_d, M, K);
    print_(D_d, K, M);

    //Expected Output:

    // 21             43             31             
    // 63             129            93             
    // 53             46             64             
    // 4              13             7  
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

    matrix::multD(S, VT, W, N, N);
    matrix::mult(U, W, A_d, 1, 1, false, false, M, N, N);
    
    print_(A_d, M, N);
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

    fit::FitGaussNewton(d_voltage, param, fit::doubleExp, N, 5);
    TestMult();
    cudaFree(d_voltage);
    free(voltage);
    
}
