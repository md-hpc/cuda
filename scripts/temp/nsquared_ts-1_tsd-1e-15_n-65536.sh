#!/bin/bash -l

#$ -N ts-1_tsd-1e-15_n-65536  # Job Name
#$ -l gpus=1
#$ -M eth@bu.edu
#$ -M ajamias@bu.edu
#$ -o output/ts-1_tsd-1e-15_n-65536_output.txt
#$ -e output/ts-1_tsd-1e-15_n-65536_output_error.txt


# specify version of CUDA to be used
module load cuda/11.3

particle_count=65536

nvcc -I./include -D TIMESTEPS=1 -D TIMESTEP_DURATION_FS=1e-15 src/pdb_importer.c src/nsquared.cu -o build/nsquared 
time build/nsquared input/random_particles-${particle_count}.pdb output/ts-1_tsd-1e-15_n-${particle_count}_output.csv


