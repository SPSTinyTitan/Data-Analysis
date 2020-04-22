#ifndef matrix_wrappers
#define matrix_wrappers

#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_tools.h>

namespace matrix{
    __host__    void    MultMat     (float* A, float* B, float* C, float alpha, float beta, int M, int N, int K);
    __host__    void    MultMatD    (float* A, float* B, float* C, int M, int N);
}

#endif //matrix_wrappers