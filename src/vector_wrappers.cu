#include "vector_wrappers.h"

namespace vector{

    //Element wise logarithm of vector.
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

    //Element wise exponentiation of vector.
    __global__ void exp_(float* A, float* B, int N){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < N)
            B[i] = expf(A[i]);
    }
    __host__ void exp(float* A, float* B, int N){
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        exp_<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
        gpuErrchk(cudaDeviceSynchronize());
    }


    //Element wise absolute value of vector.
    __global__ void abs_(float* A, float* B, int N){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < N)
            B[i] = fabsf(A[i]);
    }
    __host__ void abs(float* A, float* B, int N){
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        abs_<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
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

    //Element wise multiplication of vector.
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

    //Element wise division of vectors.
    __global__ void div_(float* A, float* B, float* C, int N){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < N)
            C[i] = A[i] / B[i];
    }
    __global__ void div_(float* A, float B, float* C, int N){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < N)
            C[i] = A[i] / B;
    }
    __global__ void div_(float A, float* B, float* C, int N){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < N)
            C[i] = A / B[i];
    }
    __global__ void dot_(float* A, float* B, float* result, int N){
        size_t result_size = threadsPerBlock * sizeof(float);
        mult_<<<gridDim, blockDim>>>(A, B, result, N);
        sum_<<<blocksPerGrid, threadsPerBlock, blocksPerGrid * result_size>>>(result, result, N);
        sum_<<<1, threadsPerBlock, result_size>>>(result, result, blocksPerGrid);
    }
    __host__ float dot(float* A, float* B, int N){
        float *result_d; gpuErrchk(cudaMalloc(&result_d, N * sizeof(float)));
        float result;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        dot_<<<blocksPerGrid, threadsPerBlock>>>(A, B, result_d, N);
        gpuErrchk(cudaMemcpy(&result, result_d, sizeof(float), cudaMemcpyDeviceToHost));
        if (result_d) cudaFree(result_d);
        return result;
    }

    __host__ void div(float* A, float* B, float* C, int N){
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        div_<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
        gpuErrchk(cudaDeviceSynchronize());
    }
    __host__ void div(float* A, float B, float* C, int N){
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        div_<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
        gpuErrchk(cudaDeviceSynchronize());
    }
    __host__ void div(float A, float* B, float* C, int N){
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        div_<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
        gpuErrchk(cudaDeviceSynchronize());
    }


    


    //Sum of elements in vector.
    __global__ void sum_(float* X, float* result, int N){
        extern __shared__ float s[];                     
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int threadsPerGrid = gridDim.x * blockDim.x;
        float sum = 0;

        //Each thread sum a portion of elements
        for (int i = idx; i < N; i += threadsPerGrid){
            sum += X[i];
        }
        
        //Sum results of each thread within each block
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
    //Applies lambda function to a vector
    // template<typename f>
    // __device__ void apply__(float* A, float* B, f&& lambda, int N){
    //     int i = blockDim.x * blockIdx.x + threadIdx.x;
    //     if (i < N)
    //         B[i] = lambda(A[i]);
    // }
    // template<typename f>
    // __device__ void apply__(float* A, float B, float* C, f&& lambda, int N){
    //     int i = blockDim.x * blockIdx.x + threadIdx.x;
    //     if (i < N)
    //         C[i] = lambda(A[i], B);
    // }
    // template<typename f>
    // __device__ void apply__(float* A, float* B, float* C, f&& lambda, int N){
    //     int i = blockDim.x * blockIdx.x + threadIdx.x;
    //     if (i < N)
    //         C[i] = lambda(A[i], B[i]);
    // }
}
