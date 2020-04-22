#include "vector_wrappers.h"

namespace vector{

    //Element wise logarithm of a vector
    __global__ void log_(float* A, float* B, int N){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < N)
            B[i] = logf(A[i]);
    }
    __host__ void log(float* A, float* B, int N){
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        log_<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
        gpuErrchk(cudaDeviceSynchronize());
    }

    //Element wise addition of vector. 
    __global__ void add_(float* A, float* B, float* C, int N){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < N)
            C[i] = A[i] + B[i];
    }
    __host__ void add(float* A, float* B, float* C, int N){
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        add_<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
        gpuErrchk(cudaDeviceSynchronize());
    }

    //Element wise subtraction of vector. 
    __global__ void sub_(float* A, float* B, float* C, int N){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < N)
            C[i] = A[i] - B[i];
    }
    __host__ void sub(float* A, float* B, float* C, int N){
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        sub_<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
        gpuErrchk(cudaDeviceSynchronize());
    }

    //Element wise multiplication of vectors. For dot products see DotVec().
    __global__ void mult_(float* A, float* B, float* C, int N){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < N)
            C[i] = A[i] * B[i];
    }
    __global__ void mult_(float* A, float B, float* C, int N){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < N)
            C[i] = A[i] * B;
    }
    //Implement this
    // __host__ float DotVecf(float* A, float* B, int N){

    // }
    __host__ void mult(float* A, float* B, float* C, int N){
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        mult_<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
        gpuErrchk(cudaDeviceSynchronize());
    }
    __host__ void mult(float* A, float B, float* C, int N){
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        mult_<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
        gpuErrchk(cudaDeviceSynchronize());
    }
    __host__ void mult(float A, float* B, float* C, int N){
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        mult_<<<blocksPerGrid, threadsPerBlock>>>(B, A, C, N);
        gpuErrchk(cudaDeviceSynchronize());
    }

    //Sum of elements in vector.
    __global__ void sum_(float* X, float* result, int N){
        extern __shared__ float s[];                     
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int threadsPerGrid = gridDim.x * blockDim.x;
        float sum = 0;

        //Having each thread sum a portion of elements
        for (int i = idx; i < N; i += threadsPerGrid){
            sum += X[i];
        }
        
        //Summing the results of each thread in each block
        float* temp = &s[gridDim.x * blockIdx.x];
        temp[threadIdx.x] = sum;
        __syncthreads();
        for (int i = blockDim.x/2; i > 0; i = i/2){
            if (threadIdx.x < i)
                temp[threadIdx.x] += temp[threadIdx.x + i];
            __syncthreads();
        }

        //The first thread in the block writes to result
        if (threadIdx.x == 0)
            result[blockIdx.x] = temp[0];
    }
    __host__ float sum(float* A, int N){
        
        size_t result_size = threadsPerBlock * sizeof(float);

        float result;
        float *d_result; cudaMalloc(&d_result, result_size);

        sum_<<<blocksPerGrid, threadsPerBlock, blocksPerGrid * result_size>>>(A, d_result, N);
        sum_<<<1, threadsPerBlock, result_size>>>(d_result, d_result, blocksPerGrid);

        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

        if (d_result) cudaFree(d_result);
        return result;
    }

    //Copy vector. A = B; 
    __global__ void copy_(float* A, float* B, int N){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < N)
            A[i] = B[i];
    }
    __host__ void copy(float* A, float* B, int N){
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        copy_<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
        gpuErrchk(cudaDeviceSynchronize());
    }

}