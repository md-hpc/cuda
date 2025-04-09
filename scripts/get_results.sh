#!/bin/bash

PROJ_DIR=$(git rev-parse --show-toplevel)

for implementation in nsquared nsquared_shared nsquared_n3l cell_list; do  
	for particle_count in 1024 4096 16384 65536; do
		cat ${PROJ_DIR}/output/${implementation}/ts-1_tsd-1e-15_n-${particle_count}_output.txt
	done
done

