#!/bin/bash

PROJ_DIR=$(git rev-parse --show-toplevel)

TIMESTEPS=300
TIMESTEP_DURATION=2.5e-13

TYPE=TIME_RUN

if [ $"{TYPE}" = "SIMULATE" ]; then
for implementation in nsquared nsquared_shared nsquared_n3l cell_list cell_list_n3l; do
	python3 ${PROJ_DIR}/src/viz.py 300 ${PROJ_DIR}/output/${implementation}/ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_1024_output.csv ${PROJ_DIR}/output/${implementation}/ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_1024_output.gif

	for particle_count in 4096 16384 65536; do
		python3 ${PROJ_DIR}/src/viz.py 1000 ${PROJ_DIR}/output/${implementation}/ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_${particle_count}_output.csv ${PROJ_DIR}/output/${implementation}/ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_${particle_count}_output.gif
	done
done
fi

if [ $"{TYPE}" = "TIME_RUN" ]; then
echo "implementation,particle_count,time (s)"
for implementation in nsquared nsquared_shared nsquared_n3l cell_list cell_list_n3l; do
	for particle_count in 1024 4096 16384 65536; do
		cat ${PROJ_DIR}/output/${implementation}/ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_${particle_count}_output.txt
	done
done
fi
