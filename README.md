# CUDA for Strong Scaling MD
Repository for all things CUDA.

## Running CUDA
CUDA can be run on the [BU Shared Computing Cluster](https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/) or through Google Colab.

### Running CUDA on the SCC
1. Login to an scc node.
2. Execute the command: `module load cuda/11.3` to load the NVIDIA sdk tools.
3. To compile your cuda code, execute the command: `nvcc <filename> -o <outfile>`
4. To run the executable, you need a GPU node. [interactive](https://www.bu.edu/tech/support/research/system-usage/running-jobs/interactive-jobs/)/[batch](https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/#job-options)




## Resources
[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

[Intro to CUDA (Oklahoma State University ECEN 4773/5793)](https://www.youtube.com/playlist?list=PLC6u37oFvF40BAm7gwVP7uDdzmW83yHPe)

[CUDA in C/C++ on the SCC](https://www.bu.edu/tech/support/research/software-and-programming/gpu-computing/cuda-c/)
