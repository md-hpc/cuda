# CUDA for Strong Scaling MD
Repository for all things CUDA.

## Running CUDA
CUDA can be run on the [BU Shared Computing Cluster](https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/) or through Google Colab.

### Running CUDA on the SCC
1. Login to an scc node.
2. Execute the command: `module load nvidia-hpc` to load the NVIDIA sdk tools.
3. To compile your cuda code, execute the command: `nvcc <filename> -o <outfile>`
4. To run the executable, you need a GPU node. [interactive](https://www.bu.edu/tech/support/research/system-usage/running-jobs/interactive-jobs/)/[batch](https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/#job-options)
  
### Running CUDA on Google Colab
1. Create a new colab notebook.
2. Change runtime to GPU.
3. Run the following commands to load the CUDA compiler to run CUDA C++ code with Jupyter Notebook.
```
!python --version
!nvcc --version
!pip install nvcc4jupyter
%load_ext nvcc4jupyter
```
4. Run code by specifying `%%cuda` at beginning of the block followed by your C++ code.
```%%cuda
#include <stdio.h>
__global__ void hello(){
  printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}
int main(){
  // numBlocks, numThreadsPerBlock
  hello<<<4, 4>>>();
  cudaDeviceSynchronize();
}
```



## Resources
[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

[Intro to CUDA (Oklahoma State University ECEN 4773/5793)](https://www.youtube.com/playlist?list=PLC6u37oFvF40BAm7gwVP7uDdzmW83yHPe)
