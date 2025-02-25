#!/bin/bash -l

# specify version of CUDA to be used
module load cuda/11.3

nvcc -I./include -D TIMESTEPS=1 -D TIMESTEP_DURATION_FS=1e-15 src/pdb_importer.c src/nsquared.cu -o build/nsquared 
time build/nsquared tests/example_input.pdb