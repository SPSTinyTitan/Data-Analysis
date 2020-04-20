
#include <cstdlib>
#include <chrono>
#include <iostream>   
#include "vector.h"

int main(){
    int N = 100000000;
    size_t size = N * sizeof(float);
    float T = 1e-9;
    
    float* voltage = (float*)malloc(size);
    voltage[0] = 0;
    // for (int i = 1; i < N; i++){
    //     voltage[i] = voltage[i-1] * 0.99998000019999866667;
    //     if(rand()%10000 == 0)
    //         voltage[i] += 1;
    // }

    for (int i = 0; i < N; i++){
        voltage[i] = i;
    }

    float* d_voltage;
    cudaMalloc(&d_voltage, size);
    cudaMemcpy(d_voltage, voltage, size, cudaMemcpyHostToDevice);

    float* result = (float*) malloc(size);
    float* d_result = vector::MultVecf(d_voltage, d_voltage, N);
    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

    float sum = vector::SumVecf(d_voltage, 100);


    for (int i = 0; i < 100; i++){
        std::cout << result[i] << '\n';
    }

    std::cout<< sum;

    cudaFree(d_voltage);
    cudaFree(d_result);
    free(result);
    free(voltage);
    
}