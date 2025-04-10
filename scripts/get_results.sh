#!/bin/bash

PROJ_DIR=$(git rev-parse --show-toplevel)

TIMESTEPS=1
TIMESTEP_DURATION=2.5e-13

echo "implementation,particle_count,time (s)"
for implementation in nsquared nsquared_shared nsquared_n3l cell_list cell_list_n3l; do
	for particle_count in 1024 4096 16384 65536; do
		cat ${PROJ_DIR}/output/${implementation}/ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_${particle_count}_output.txt
	done
done

