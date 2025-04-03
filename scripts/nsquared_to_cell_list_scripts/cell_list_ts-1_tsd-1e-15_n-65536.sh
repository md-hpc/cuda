#!/bin/bash -l

#$ -N ts-1_tsd-1e-15_n-65536  # Job Name
#$ -l gpus=1
#$ -l gpu_type=A40 
#$ -M eth@bu.edu
#$ -M ajamias@bu.edu
#$ -o output/cell_list/ts-1_tsd-1e-15_n-65536_output.txt
#$ -e output/cell_list/ts-1_tsd-1e-15_n-65536_output_error.txt
	
module load cuda/

nvcc -I./include -D TIMESTEPS=1 -D TIMESTEP_DURATION_FS=1e-15 -D CELL_CUTOFF_RADIUS_ANGST=100 -D UNIVERSE_LENGTH=1000 -D TIME_RUN -D CELL_LENGTH_X=10 -D CELL_LENGTH_Y=10 -D CELL_LENGTH_Z=10 src/pdb_importer.c src/cell_list.cu -o build/cell_list
build/cell_list input/random_particles-65536.pdb output/cell_list/ts-1_tsd-1e-15_n-65536_output.csv
build/cell_list input/random_particles-65536.pdb output/cell_list/ts-1_tsd-1e-15_n-65536_output.csv

