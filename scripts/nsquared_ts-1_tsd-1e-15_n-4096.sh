#!/bin/bash -l

#$ -N ts-1_tsd-1e-15_n-4096  # Job Name
#$ -l gpus=1
#$ -l gpu_type=A40 
#$ -M eth@bu.edu
#$ -M ajamias@bu.edu
#$ -o output/nsquared/ts-1_tsd-1e-15_n-4096_output.txt
#$ -e output/nsquared/ts-1_tsd-1e-15_n-4096_output_error.txt
	
module load cuda/11.3

nvcc -I./include -D TIMESTEPS=1 -D TIMESTEP_DURATION_FS=1e-15 -D CELL_CUTOFF_RADIUS_ANGST=10 -D UNIVERSE_LENGTH=100 -D TIME_RUN -D CELL_LENGTH_X=10 CELL_LENGTH_Y=10 CELL_LENGTH_Z=10 src/pdb_importer.c src/nsquared.cu -o build/nsquared
time build/nsquared input/random_particles-4096.pdb output/nsquared/ts-1_tsd-1e-15_n-4096_output.csv

