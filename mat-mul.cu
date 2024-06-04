#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

double parallelMatrixMultiply(float *A, float *B, float *C, int N, int blockSize) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    matrixMultiplyKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Elapsed time GPU: %f ms\n", elapsedTime);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return elapsedTime;
}

int main() {
    int N = 2048; // Change as needed
    int blockSize = 16; // Change as needed

    printf("Masukan N dan blockSize\n");
    scanf("%d %d", &N, &blockSize);
    printf("N: %d, Block size: %d\n", N, blockSize);

    float *A = (float*)malloc(N * N * sizeof(float));
    float *B = (float*)malloc(N * N * sizeof(float));
    float *C = (float*)malloc(N * N * sizeof(float));

    // Initialize matrices A and B
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    clock_t begin = clock();

    double gpuTime;

    gpuTime = parallelMatrixMultiply(A, B, C, N, blockSize);

    clock_t end = clock();
    double cpuTime = (double)(end-begin)/(CLOCKS_PER_SEC/1000);

    printf("Elapsed time on CPU: %f ms\n", cpuTime);

    printf("COMM TIME = CPU-GPU = %f ms\n", cpuTime-gpuTime);

    printf("x/y = %f\n", gpuTime/(cpuTime-gpuTime));

    printf("SMALL PART OF MATRIX:\n");

    // Print a small part of the matrix to verify correctness
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%f ", C[i * N + j]);
        }
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
