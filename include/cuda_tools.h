#pragma once
#ifndef cuda_tools
#define cuda_tools

#include <cstdio>
#include <iostream>
#include <iomanip>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) 
            exit(code);
    }
}
__host__ void print_(const float* A_d, int M, int N);
__host__ void print(const float* A_h, int M, int N);

#endif //cuda_tools