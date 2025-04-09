#!/bin/bash -l

#$ -N ts-1_tsd-1e-15_n-3000  # Job Name
#$ -l gpus=1
#$ -l gpu_type=A40 
#$ -M eth@bu.edu
#$ -M ajamias@bu.edu
#$ -o output/nsquared/ts-1_tsd-1e-15_n-3000_output.txt
#$ -e output/nsquared/ts-1_tsd-1e-15_n-3000_output_error.txt
	
module load cuda/12.5

nvcc -I./include -D TIMESTEPS=1 -D TIMESTEP_DURATION_FS=2.5e-13 -D UNIVERSE_LENGTH=30 -D TIME_RUN src/pdb_importer.c src/nsquared.cu -o build/nsquared
build/nsquared input/n-3000.pdb output/nsquared/ts-1_tsd-1e-15_n-3000_output.csv

