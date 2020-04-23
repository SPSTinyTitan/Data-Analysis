#include "cuda_tools.h"


__host__ void print_(const float* A_d, int M, int N){
    float* A_h = (float*)malloc(M * N * sizeof(float));
    gpuErrchk(cudaMemcpy(A_h, A_d, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M; i++){
        std::cout << "{";
        for (int j = 0; j < N; j++){
            std::cout << std::left << std::setw(15) << A_h[i + j * M] << ",";
        }
        std::cout << "}\n";
    }
}
__host__ void print(const float* A_h, int M, int N){
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            std::cout << std::left << std::setw(15) << A_h[i + j * M];
        }
        std::cout << '\n';
    }
}    