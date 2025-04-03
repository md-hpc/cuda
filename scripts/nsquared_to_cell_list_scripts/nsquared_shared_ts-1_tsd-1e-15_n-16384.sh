#!/bin/bash -l

#$ -N ts-1_tsd-1e-15_n-16384  # Job Name
#$ -l gpus=1
#$ -l gpu_type=A40 
#$ -M eth@bu.edu
#$ -M ajamias@bu.edu
#$ -o output/nsquared_shared/ts-1_tsd-1e-15_n-16384_output.txt
#$ -e output/nsquared_shared/ts-1_tsd-1e-15_n-16384_output_error.txt
	
module load cuda/

nvcc -I./include -D TIMESTEPS=1 -D TIMESTEP_DURATION_FS=1e-15 -D CELL_CUTOFF_RADIUS_ANGST=100 -D UNIVERSE_LENGTH=1000 -D TIME_RUN -D CELL_LENGTH_X=10 -D CELL_LENGTH_Y=10 -D CELL_LENGTH_Z=10 src/pdb_importer.c src/nsquared_shared.cu -o build/nsquared_shared
build/nsquared_shared input/random_particles-16384.pdb output/nsquared_shared/ts-1_tsd-1e-15_n-16384_output.csv
build/nsquared_shared input/random_particles-16384.pdb output/nsquared_shared/ts-1_tsd-1e-15_n-16384_output.csv

