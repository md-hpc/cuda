#!/bin/bash

PROJ_DIR=$(git rev-parse --show-toplevel)

TIMESTEPS=300
TIMESTEP_DURATION=2.5e-13

for implementation in nsquared nsquared_shared nsquared_n3l cell_list cell_list_n3l; do
	rm -rf ${PROJ_DIR}/output/${implementation}/*

	for particle_count in 1024 4096 16384 65536; do
		qsub ${PROJ_DIR}/scripts/${implementation}_ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_${particle_count}.sh
	done
done

