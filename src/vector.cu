#include "vector.h"

namespace vector{
    //Element wise logarithm of a vector
    __global__ void LogVecf(float* A, float* B, int N){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < N)
            B[i] = logf(A[i]);
    }
    __host__ float* LogVec(float* A, int N){
        float* B;
        cudaMalloc(&B, N*sizeof(float));
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        LogVecf<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
        return B;
    }

    //Element wise addition of vector. 
    __global__ void AddVecf(float* A, float* B, float* C, int N){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < N)
            C[i] = A[i] + B[i];
    }
    __host__ float* AddVecf(float* A, float* B, int N){
        float* C;
        cudaMalloc(&C, N*sizeof(float));
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        AddVecf<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
        return C;
    }

    //Element wise multiplication of vectors. For dot products see DotVec().
    __global__ void MultVecf(float* A, float* B, float* C, int N){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < N)
            C[i] = A[i] * B[i];
    }
    __host__ float* MultVecf(float* A, float* B, int N){
        float* C;
        cudaMalloc(&C, N*sizeof(float));
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        MultVecf<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
        return C;
    }
    __host__ float DotVecf(float* A, float* B, int N){
        float* C = MultVecf(A, B, N);
        float result = SumVecf(C, N);
        cudaFree(C);
        return result;
    }

    //Sum of elements in vector.
    __global__ void SumVecf(float* X, float* result, int N){
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
    __host__ float SumVecf(float* A, int N){
        float result;
        float *d_result;
        size_t result_size = threadsPerBlock * sizeof(float);
        cudaMalloc(&d_result, result_size);
        SumVecf<<<blocksPerGrid, threadsPerBlock, blocksPerGrid * result_size>>>(A, d_result, N);
        SumVecf<<<1, threadsPerBlock, result_size>>>(d_result, d_result, blocksPerGrid);
        cudaDeviceSynchronize();
        cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        return result;
    }
}