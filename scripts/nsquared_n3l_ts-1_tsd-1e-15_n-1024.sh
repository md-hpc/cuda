#!/bin/bash -l

#$ -N ts-1_tsd-1e-15_n-1024  # Job Name
#$ -l gpus=1
#$ -l gpu_type=A40 
#$ -M eth@bu.edu
#$ -M ajamias@bu.edu
#$ -o output/nsquared_n3l/ts-1_tsd-1e-15_n-1024_output.txt
#$ -e output/nsquared_n3l/ts-1_tsd-1e-15_n-1024_output_error.txt
	
module load cuda/11.3

nvcc -I./include -D TIMESTEPS=1 -D TIMESTEP_DURATION_FS=2.5e-13 -D UNIVERSE_LENGTH=300 -D TIME_RUN src/pdb_importer.c src/nsquared_n3l.cu -o build/nsquared_n3l
time build/nsquared_n3l input/random_particles-1024.pdb output/nsquared_n3l/ts-1_tsd-1e-15_n-1024_output.csv

