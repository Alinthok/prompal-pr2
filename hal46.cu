#include <iostream>
#include <cuda_runtime.h>

__global__ void kernel( int *a )
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    a[idx] = 7;
}

int main() {
    int N = 15;
    int *a, *h_a;

    cudaMalloc(&a, N * sizeof(int));
    h_a = (int *)malloc(N * sizeof(int));

    kernel<<<3, 5>>>(a);

    cudaMemcpy(h_a, a, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int ii = 0; ii < N; ++ii) {
        printf("%d ", h_a[ii]);
    }

    return 0;
}