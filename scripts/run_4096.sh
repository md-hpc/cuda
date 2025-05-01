#!/bin/bash

PROJ_DIR=$(git rev-parse --show-toplevel)

echo "implementation,N,time (s)"

for particle_count in particle_counts; do
	for implementation in nsquared nsquared_shared nsquared_n3l cell_list cell_list_n3l; do
		${PROJ_DIR}/build/${implementation} ${PROJ_DIR}/input/random_particles-4096.pdb ${PROJ_DIR}/output/${implementation}/ts_1_tsd_2.5e-13_n_4096_output.csv
	done 
done
