#!/bin/bash -l

#$ -N ts-1_tsd-1e-15_n-16384  # Job Name
#$ -l gpus=1
#$ -l gpu_type=A40 
#$ -M eth@bu.edu
#$ -M ajamias@bu.edu
#$ -o output/nsquared_shared/ts-1_tsd-1e-15_n-16384_output.txt
#$ -e output/nsquared_shared/ts-1_tsd-1e-15_n-16384_output_error.txt
	
module load cuda/11.3

nvcc -I./include -D TIMESTEPS=1 -D TIMESTEP_DURATION_FS=2.5e-13 -D UNIVERSE_LENGTH=1000 -D TIME_RUN src/pdb_importer.c src/nsquared_shared.cu -o build/nsquared_shared
time build/nsquared_shared input/random_particles-16384.pdb output/nsquared_shared/ts-1_tsd-1e-15_n-16384_output.csv

