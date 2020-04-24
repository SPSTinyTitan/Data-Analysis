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

    //Element wise multiplication of vectors.
    __global__  void    mult_   (float* A, float* B, float* C, int N);
    __global__  void    mult_   (float* A, float B, float* C, int N);
    __host__    void    mult    (float* A, float* B, float* C, int N);
    __host__    void    mult    (float* A, float B, float* C, int N);
    __host__    void    mult    (float A, float* B, float* C, int N);

    //Element wise division of vectors.
    __global__  void    div_    (float* A, float* B, float* C, int N);
    __global__  void    div_    (float* A, float B, float* C, int N);
    __global__  void    div_    (float A, float* B, float* C, int N);
    __host__    void    div     (float* A, float* B, float* C, int N);
    __host__    void    div     (float* A, float B, float* C, int N);
    __host__    void    div     (float A, float* B, float* C, int N);

    //Dot product
    __global__  void    dot_    (float* A, float* B, float* result, int N);
    __host__    float   dot     (float* A, float* B, int N);

    //Sum of elements in vector.
    __global__  void    sum_    (float* X, float* result, int N);
    __host__    float   sum     (float* A, int N);

    //Copy vector. A = B;
    __global__  void    copy_   (float* A, float* B, int N);
    __host__    void    copy    (float* A, float* B, int N);

    //Applies Lambda function
    // template<typename f>
    // __device__  void    apply__ (float* A, float* B, f&& lambda, int N);
    // template<typename f>
    // __device__  void    apply__ (float* A, float* B, float* C, f&& lambda, int N);
    // template<typename f>
    // __device__  void    apply__ (float* A, float B, float* C, f&& lambda, int N);

    //Logic operators
    // __global__  void    g_      (float* A, float* B, float* C, int N);
    // __global__  void    g_      (float* A, float B, float* C, int N);
    // __global__  void    l_      (float* A, float* B, float* C, int N);
    // __global__  void    l_      (float* A, float* B, float* C, int N);
    // __global__  void    geq_    (float* A, float* B, float* C, int N);
    // __global__  void    geq_    (float* A, float B, float* C, int N);
    // __global__  void    leq_    (float* A, float* B, float* C, int N);
    // __global__  void    leq_    (float* A, float* B, float* C, int N);
    // __global__  void    eq_     (float* A, float* B, float* C, int N);
    // __global__  void    eq_     (float* A, float B, float* C, int N);
    // __global__  void    eq_     (float* A, float* B, float* C, int N);
    // __global__  void    eq_     (float* A, float* B, float* C, int N);
    // __host__    void    g       (float* A, float* B, float* C, int N);
    // __host__    void    g       (float* A, float B, float* C, int N);
    // __host__    void    l       (float* A, float* B, float* C, int N);
    // __host__    void    l       (float* A, float* B, float* C, int N);
    // __host__    void    geq     (float* A, float* B, float* C, int N);
    // __host__    void    geq     (float* A, float B, float* C, int N);
    // __host__    void    leq     (float* A, float* B, float* C, int N);
    // __host__    void    leq     (float* A, float* B, float* C, int N);
    // __host__    void    eq      (float* A, float* B, float* C, int N);
    // __host__    void    eq      (float* A, float B, float* C, int N);
    // __host__    void    eq      (float* A, float* B, float* C, int N);
    // __host__    void    eq      (float* A, float* B, float* C, int N);
}
#endif //vector_wrappers
