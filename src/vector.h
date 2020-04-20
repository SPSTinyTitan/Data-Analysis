namespace vector{
    const int threadsPerBlock = 256;
    const int blocksPerGrid = 20;

    __global__ void LogVecf(float* A, float* B, int N);
    __global__ void AddVecf(float* A, float* B, float* C, int N);
    __global__ void MultVecf(float* A, float* B, float* C, int N);
    __global__ void SumVecf(float* X, float* result, int N);
    __global__ void ScaleVecf(float* A, float B, float* C, int N);
    __global__ void SubVecf(float* A, float* B, float* C, int N);
    __host__ float* LogVec(float* A, int N);
    __host__ float* AddVecf(float* A, float* B, int N);
    __host__ float* MultVecf(float* A, float* B, int N);
    __host__ float DotVecf(float* A, float* B, int N);
    __host__ float SumVecf(float* A, int N);
    __host__ float* ScaleVecf(float* A, float B, int N);
    __host__ float* SubVecf(float* A, float* B, int N);
}