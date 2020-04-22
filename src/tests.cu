#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) 
            exit(code);
    }
}

void TestMult(){
    int M = 2;
    int N = 4;
    int K = 3;
    float A[8] = {3, 9, 2, 1, 1, 3, 5, 0};
    float B[12] = {2, 1, 2, 8, 9, 3, 4 ,1, 0, 5, 7, 5};
    float* C = (float*)malloc(M * K * sizeof(float));
    float* A_d; gpuErrchk(cudaMalloc(&A_d, M * N * sizeof(float)));
    float* B_d; gpuErrchk(cudaMalloc(&B_d, N * K * sizeof(float)));
    float* C_d; gpuErrchk(cudaMalloc(&C_d, M * K * sizeof(float)));
    cudaMemcpy(A_d, &A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, &B, N * K * sizeof(float), cudaMemcpyHostToDevice);
    matrix::MultMath(A_d, B_d, C_d, M, N, K);
    
    cudaMemcpy(C, C_d, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    

    matrix::Print(C, M, K);
}


void TestSVD(){
    int M = 3;
    int N = 2;
    float A[20] = {
        2,0,0,
        0,-3,0};
    float* J;
    gpuErrchk(cudaMalloc(&J, M * N * sizeof(float)));
    cudaMemcpy(J, &A, M * N * sizeof(float), cudaMemcpyHostToDevice);

    int P = M;
    if (M > N)
        P = N;

    float* Sh =     (float*)malloc(P * sizeof(float));
    float* Uh =     (float*)malloc(M * M * sizeof(float));
    float* VTh =    (float*)malloc(N * N * sizeof(float));
    float* S;       gpuErrchk(cudaMalloc(&S, P * sizeof(float)));
    float* U;       gpuErrchk(cudaMalloc(&U, M * M * sizeof(float)));
    float* VT;      gpuErrchk(cudaMalloc(&VT, N * N * sizeof(float)));

    vector::SVD(J, M, N, S, U, VT);
    
    cudaMemcpy(Sh, S, P * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Uh, U, M * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(VTh, VT, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++){
        for (int j = 0; j < M; j++){
            std::cout << std::left << std::setw(15) << Uh[i + j * M];
        }
        std::cout << '\n';
    }
    for (int i = 0; i < P; i++){
        std::cout << std::left << std::setw(15) << Sh[i];
    }
    std::cout << '\n';
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            std::cout << std::left << std::setw(15) << VTh[i + j * N];
        }
        std::cout << '\n';
    }

    float* Sm = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * N; i++){
        Sm[i] = 0;
    }
    for (int i = 0; i < P; i++){
        Sm[i*M + i] = Sh[i];
    }
    cudaMemcpy(J, Sm, M * N * sizeof(float), cudaMemcpyHostToDevice);

    float* US;       gpuErrchk(cudaMalloc(&US, M * N * sizeof(float)));

    matrix::MultMath(J, U, US, M, M, N);
    matrix::MultMath(US, VT, J, M, N, N);

    matrix::Print(J, M, N);
}
