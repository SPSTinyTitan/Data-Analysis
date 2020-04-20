
//Numerical derivative y'(t2) = y(t2) - y(t1).
//Pad at start y'(0) = y'(1)
__global__ void Deriv(float* A, float* B, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
    if (i < N)
        B[i] = A[i] - A[i - 1];
}
__host__ float* Deriv(float* A, int N){
    float* B;
    gpuErrchk(cudaMalloc(&B, N*sizeof(float)));
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    Deriv<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    B[0] = B[1];
    return B;
}
__host__ float* Deriv2(float* A, int N){
    float* B = Deriv(A,N);
    float* C = Deriv(B,N);
    C[0] = C[2];
    C[1] = C[2];
    cudaFree(B);
    return C;
}

//Algorithm obatined from Alastair Bateman.
//2 sided deltas can be taken to improve accuracy
//Though at most sample rates, this is probably unnecessary. 

__host__ void DoubleExpFit(float* A, float* param, int N){
    float* dA = Deriv(A, N);
    float* d2A = Deriv2(A, N);
    float* ASquared = MultVecf(A, A, N);
    float* dASquared = MultVecf(dA, dA, N);
    float* AdA = MultVecf(A, dA, N);
    float* Ad2A = MultVecf(A, d2A, N);
    float* dAd2A = MultVecf(dA, d2A, N);

    float ASquaredSum = SumVecf(ASquared, N);
    float dASquaredSum = SumVecf(dASquared, N);
    float AdASum = SumVecf(AdA, N);
    float Ad2ASum = SumVecf(Ad2A, N);
    float dAd2ASum = SumVecf(dAd2A, N);

    cudaFree(dA);
    cudaFree(d2A);
    cudaFree(ASquared);
    cudaFree(dASquared);
    cudaFree(AdA);
    cudaFree(Ad2A);
    cudaFree(dAd2A);

    float denom = ASquaredSum * dASquaredSum - AdASum * AdASum;
    float zNum = AdASum * Ad2ASum - ASquaredSum * dAd2ASum;
    float nNum = dAd2ASum * AdASum - Ad2ASum * dASquaredSum;
    float z = zNum/denom;
    float n = nNum/denom;
    float beta = (-z + sqrtf(z * z - 4 * n)) / 2;
    float alpha = n / beta;
    
    float k = logf(1-alpha);
    float q = logf(1-beta);
}