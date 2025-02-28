
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
//pretty sure this is only c and not c++ at the moment
//do not know if you can do c++ things


__global__ void mult2(int* Q, int* P, int *N, int size)
{
    int x = blockIdx.x + threadIdx.x;
    int y = blockIdx.y + threadIdx.y;
    printf("my thread id is x: %d and y: %d\n", x, y);
    int dotProd = 0;
    for (int f = 0; f < size; f++)
    {
        dotProd += Q[y * blockDim.x + f] * P[f * blockDim.x + x];
    }
    N[y * blockDim.x + x] = dotProd;

}

int main()
{
    const int rows = 12; //define rows
    const int colloms = 12; //define colloms
    const int ARRAY_SIZE = rows * colloms;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

    //Declare and initialize CPU arrays
    int h_A[rows][colloms];//h stands for hsot
    int h_B[rows][colloms];
    int h_out[rows][colloms];
    for (int i = 1; i < rows+1; i++)
    {
        for (int j = 1; j < rows+1; j++)
        {
            h_A[i-1][j-1] = i+j;
            h_B[i-1][j-1] = i+j;
        }
    }
    printf("\n");
    for (int i = 0; i < rows; i++)
    {
        for (int k = 0; k < colloms; k++)
        {
            printf("%d   ", h_A[i][k]);
        }
        printf("\n");
    }
    int* d_A; 
    int* d_B;
    int* d_out;
    cudaMalloc((void**)&d_A, ARRAY_BYTES); 
    cudaMalloc((void**)&d_B, ARRAY_BYTES);
    cudaMalloc((void**)&d_out, ARRAY_BYTES); //arbatrary pointer to a pointer

    //copy array to gpu
    cudaMemcpy(d_A, h_A, ARRAY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, ARRAY_BYTES, cudaMemcpyHostToDevice);
    //launch the kernel
    dim3 blockDim = 1;
    dim3 size = { rows, colloms };
    mult2 <<<blockDim, size >>> (d_A, d_B,d_out, rows); //the kernel launch



    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    //this is just a tester thing to show that there is a value
    for (int i = 0; i < rows; i++)
    {
        for (int k = 0; k < colloms; k++)
        {
            printf("%d   ", h_out[i][k]);
        }
        printf("\n");
    }
    //clean up the code
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);

    //end
    return 0;

}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }



    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
