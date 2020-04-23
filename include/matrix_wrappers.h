#ifndef matrix_wrappers
#define matrix_wrappers

#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuda_tools.h"
#include "vector_wrappers.h"

namespace matrix{
    const int BLOCK_DIM = 3;
    __host__    void    mult        (float* A, float* B, float* C, float alpha, float beta, bool TA, bool TB, int M, int N, int K);
    __host__    void    multD       (float* A, float* B, float* C, int M, int N);
    __global__  void    transpose_  (float *A, float *B, int M, int N);
    __host__    void    transpose   (float* A, float* B, int M, int N);       
}

#endif //matrix_wrappers