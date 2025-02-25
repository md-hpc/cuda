#!/bin/bash -l

#$ -N ts-1_tsd-1e-15  # Job Name
#$ -l gpus=1
#$ -M eth@bu.edu
#$ -M ajamias@bu.edu

# specify version of CUDA to be used
module load cuda/11.3

nvcc -I./include -D TIMESTEPS=1 -D TIMESTEP_DURATION_FS=1e-15 src/pdb_importer.c src/nsquared.cu -o build/nsquared 
time build/nsquared tests/example_input.pdb tests/ts-1_tsd-1e-15_output.csv > ts-1_tsd-1e-15-output.txt

