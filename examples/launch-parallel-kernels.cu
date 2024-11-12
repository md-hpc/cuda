#include <stdio.h>

/*
 * Example that launches parallel kernels.
 */

__global__ void firstParallel()
{
    // global means function will run on the GPU, but can be invoked globally (also by the CPU)
    // functions with __global__ keyword are required to return type void

  printf("This should be running in parallel.\n");
}

int main()
{
    // configures 5 thread blocks with each having 5 threads
    // <<< NUMBER_OF_BLOCKS, NUMBER_OF_THREADS_PER_BLOCK>>>
    firstParallel<<<5, 5>>>();

    // causes the host (CPU) code to wait until the device (GPU) code completes, and only then resume execution on the CPU
    cudaDeviceSynchronize();
}
