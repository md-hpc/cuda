#!/bin/bash -l

#$ -N ts-1_tsd-1e-15_n-65536  # Job Name
#$ -l gpus=1
#$ -M eth@bu.edu
#$ -M ajamias@bu.edu
#$ -o output/nsquared_shared/ts-1_tsd-1e-15_n-65536_output.txt
#$ -e output/nsquared_shared/ts-1_tsd-1e-15_n-65536_output_error.txt
	
module load cuda/11.3

nvcc -I./include -D TIMESTEPS=1 -D TIMESTEP_DURATION_FS=1e-15 src/pdb_importer.c src/nsquared_shared.cu -o build/nsquared_shared
time build/nsquared_shared input/random_particles-65536.pdb output/nsquared_shared/ts-1_tsd-1e-15_n-65536_output.csv

