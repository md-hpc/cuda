#!/bin/bash -l

#$ -N ts-1_tsd-1e-15_n-1024  # Job Name
#$ -l gpus=1
#$ -M eth@bu.edu
#$ -M ajamias@bu.edu
#$ -o output/nsquared_shared/ts-1_tsd-1e-15_n-1024_output.txt
#$ -e output/nsquared_shared/ts-1_tsd-1e-15_n-1024_output_error.txt
	
module load cuda/11.3

nvcc -I./include -D TIMESTEPS=1 -D TIMESTEP_DURATION_FS=1e-15 src/pdb_importer.c src/nsquared_shared.cu -o build/nsquared_shared
time build/nsquared_shared input/random_particles-1024.pdb output/nsquared_shared/ts-1_tsd-1e-15_n-1024_output.csv

