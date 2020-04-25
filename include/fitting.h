#ifndef fitting
#define fitting

#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "cuda_tools.h"
#include "vector_wrappers.h"
#include "matrix_wrappers.h"

namespace fit{

    const int threadsPerBlock = 256;
    const int blocksPerGrid = 20;

    //Return: ae^(kt) + be^(qt) + c
    __global__  void    doubleExp_  (float* A, float a, float b, float c, float k, float q, int N);
    __host__    void    doubleExp   (float* A, float* param, int N);

    //Return: at + b
    __global__  void    linear_     (float* A, float a, float b, int N);
    __host__    void    linear      (float* A, float* param, int N);

    //Returns coef matrix with first column being t_n and second column constant
    __global__  void    lincoef_    (float* A, int N);
    __host__    void    lincoef     (float* A, int N);


    //Calculates the Jacobian of f(t) for K parameters and N data points
    //Returns N x K matrix in column major order
    __host__    void    jacobian    (float* J, void f(float* A, float* param, int N), float* param, int N, int K);


    //Same as jacboian() but uses adaptive values of epsilon scaled with the value of param.
    //This can achieve better precisions across a wider range of values.
    //TODO: Consider logarithms then addition instead of multiplications. Reduces arithmetic error.
    __host__    void    jacobian_v2 (float* J, void f(float* A, float* param, int N), float* param, int N, int K);

    __host__    void    svd         (float* A, float* U, float* S, float* VT, int M, int N);
}

#endif //fitting