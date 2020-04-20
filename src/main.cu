
#include <cstdlib>
#include <chrono>
#include <iostream>   
#include "vector.h"
namespace vector {
//Return: ae^(kt) + be^(qt) + c
__global__ void DoubleExp(float* A, float a, float b, float c, float k, float q, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        A[i] = a * expf(k * i) + b * expf(q * i) + c;
}
__host__ float* DoubleExp(float* param, int N){
    float* A;
    cudaMalloc(&A, N*sizeof(float));
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    DoubleExp<<<blocksPerGrid, threadsPerBlock>>>(A, param[0], param[1], param[2], param[3], param[4], N);
    return A;
}

//Numerical derivative y'(t2) = y(t2) - y(t1).
//Pad at start y'(0) = y'(1)
__global__ void Deriv(float* A, float* B, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
    if (i < N)
        B[i] = A[i] - A[i - 1];
}
__host__ float* Deriv(float* A, int N){
    float* B;
    cudaMalloc(&B, N*sizeof(float));
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    Deriv<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    B[0] = B[1];
    return B;
}
__host__ float* Deriv2(float* A, int N){
    float* B = Deriv(A,N);
    float* C = Deriv(B,N);
    C[0] = C[2];
    C[1] = C[2];
    cudaFree(B);
    return C;
}

//Algorithm obatined from Alastair Bateman.
//2 sided deltas can be taken to improve accuracy
//Though at most sample rates, this is probably unnecessary. 

__host__ void DoubleExpFit(float* A, float* param, int N){
    float* dA = Deriv(A, N);
    float* d2A = Deriv2(A, N);
    float* ASquared = MultVecf(A, A, N);
    float* dASquared = MultVecf(dA, dA, N);
    float* AdA = MultVecf(A, dA, N);
    float* Ad2A = MultVecf(A, d2A, N);
    float* dAd2A = MultVecf(dA, d2A, N);

    float ASquaredSum = SumVecf(ASquared, N);
    float dASquaredSum = SumVecf(dASquared, N);
    float AdASum = SumVecf(AdA, N);
    float Ad2ASum = SumVecf(Ad2A, N);
    float dAd2ASum = SumVecf(dAd2A, N);

    cudaFree(dA);
    cudaFree(d2A);
    cudaFree(ASquared);
    cudaFree(dASquared);
    cudaFree(AdA);
    cudaFree(Ad2A);
    cudaFree(dAd2A);

    float denom = ASquaredSum * dASquaredSum - AdASum * AdASum;
    float zNum = AdASum * Ad2ASum - ASquaredSum * dAd2ASum;
    float nNum = dAd2ASum * AdASum - Ad2ASum * dASquaredSum;
    float z = zNum/denom;
    float n = nNum/denom;
    float beta = (-z + sqrtf(z * z - 4 * n)) / 2;
    float alpha = n / beta;
    
    float k = logf(1-alpha);
    float q = logf(1-beta);
}

//Calculates the Jacobian of f for K parameters and N data points
const float eps = 1e-8;
__host__ float* Jacobian(float* f(float* param, int N), float* param, int N, int K){
    float* jacob;
    cudaMalloc(&jacob, N * K * sizeof(float));

    float* prev = f(param, N);
    float* next;
    for (int i = 0; i < K; i++){
        param[i] += eps;
        next = f(param, N);
        jacob[i * N] = SubVecf(next, prev, N);
        param[i] -= eps;
        cudaFree(next);
    }
    cudaFree(prev);
    return ScaleVecf(jacob, (1./eps));
}

//Gauss-Newton method
__host__ void FitGaussNewton(float* A, float* param, float* f(float* param, int N), int N, int K){
    
}

}





int main(){
    int N = 100000;
    size_t size = N * sizeof(float);
    float T = 1e-9;
    float* voltage = (float*)malloc(size);
    float* d_voltage;
    cudaMalloc(&d_voltage, size);

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
    float param[5] = {1,-1,1,-1 * 1e-3,-10 * 1e-3};
    float* result = (float*) malloc(size);
    float* d_result = vector::Jacobian(vector::DoubleExp(),param, N, 5);
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 1000; i++){
        std::cout << result[i] << '\n';
    }

    cudaFree(d_voltage);
    cudaFree(d_result);
    free(result);
    free(voltage);
    
}