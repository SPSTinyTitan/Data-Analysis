#ifndef vector_wrappers
#define vector_wrappers

#include "cuda_tools.h"

namespace vector{
    const int threadsPerBlock = 256;
    const int blocksPerGrid = 20;

    //Element wise logarithm of a vector
    __global__  void    log_    (float* A, float* B, int N);
    __host__    void    log     (float* A, float* B, int N);

    //Element wise addition of vector. 
    __global__  void    add_    (float* A, float* B, float* C, int N);
    __host__    void    add     (float* A, float* B, float* C, int N);

    //Element wise subtraction of vector. 
    __global__  void    sub_    (float* A, float* B, float* C, int N);
    __host__    void    sub     (float* A, float* B, float* C, int N);

    //Element wise multiplication of vectors. For dot products see DotVec().
    __global__  void    mult_   (float* A, float* B, float* C, int N);
    __global__  void    mult_   (float* A, float B, float* C, int N);
    __host__    void    mult    (float* A, float* B, float* C, int N);
    __host__    void    mult    (float* A, float B, float* C, int N);
    __host__    void    mult    (float A, float* B, float* C, int N);

    //Implement this
    // __host__ float DotVecf(float* A, float* B, int N);

    //Sum of elements in vector.
    __global__  void    sum_    (float* X, float* result, int N);
    __host__    float   sum     (float* A, int N);

    //Copy vector.
    __global__  void    copy_   (float* A, float* B, int N);
    __host__    void    copy    (float* A, float* B, int N);
}

#endif //vector_wrappers