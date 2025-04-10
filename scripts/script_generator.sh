#!/bin/bash

PROJ_DIR=$(git rev-parse --show-toplevel)
SCRIPTS_DIR=${PROJ_DIR}/scripts

TIMESTEPS=1
TIMESTEP_DURATION=2.5e-13

rm -rf ${SCRIPTS_DIR}/nsquared*.sh
rm -rf ${SCRIPTS_DIR}/cell_list*.sh

# nsquared implementations
for implementation in nsquared nsquared_shared nsquared_n3l; do
	cat > ${SCRIPTS_DIR}/${implementation}_ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_1024.sh <<EOF
#!/bin/bash -l

#$ -N ${implementation}_n_1024  # Job Name
#$ -l gpus=1
#$ -l gpu_type=A40 
#$ -M eth@bu.edu
#$ -M ajamias@bu.edu
#$ -o ${PROJ_DIR}/output/${implementation}/ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_1024_output.txt
#$ -e ${PROJ_DIR}/output/${implementation}/ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_1024_output_error.txt
	
module load cuda/11.3

nvcc -I./include -D TIMESTEPS=1 -D TIMESTEP_DURATION_FS=2.5e-13 -D UNIVERSE_LENGTH=300 -D TIME_RUN ${PROJ_DIR}/src/pdb_importer.c ${PROJ_DIR}/src/${implementation}.cu -o ${PROJ_DIR}/build/${implementation}
${PROJ_DIR}/build/${implementation} ${PROJ_DIR}/input/random_particles-1024.pdb ${PROJ_DIR}/output/${implementation}/ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_1024_output.csv
EOF

	for particle_count in 4096 16384 65536; do
		cat > ${SCRIPTS_DIR}/${implementation}_ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_${particle_count}.sh <<EOF
#!/bin/bash -l

#$ -N ${implementation}_n_${particle_count}  # Job Name
#$ -l gpus=1
#$ -l gpu_type=A40 
#$ -M eth@bu.edu
#$ -M ajamias@bu.edu
#$ -o ${PROJ_DIR}/output/${implementation}/ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_${particle_count}_output.txt
#$ -e ${PROJ_DIR}/output/${implementation}/ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_${particle_count}_output_error.txt
	
module load cuda/11.3

nvcc -I./include -D TIMESTEPS=1 -D TIMESTEP_DURATION_FS=2.5e-13 -D UNIVERSE_LENGTH=1000 -D TIME_RUN ${PROJ_DIR}/src/pdb_importer.c ${PROJ_DIR}/src/${implementation}.cu -o ${PROJ_DIR}/build/${implementation}
${PROJ_DIR}/build/${implementation} ${PROJ_DIR}/input/random_particles-${particle_count}.pdb ${PROJ_DIR}/output/${implementation}/ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_${particle_count}_output.csv
EOF

	done
done

# cell list implementations
for implementation in cell_list cell_list_n3l; do
	cat > ${SCRIPTS_DIR}/${implementation}_ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_1024.sh <<EOF
#!/bin/bash -l

#$ -N ${implementation}_n_1024  # Job Name
#$ -l gpus=1
#$ -l gpu_type=A40 
#$ -M eth@bu.edu
#$ -M ajamias@bu.edu
#$ -o ${PROJ_DIR}/output/${implementation}/ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_1024_output.txt
#$ -e ${PROJ_DIR}/output/${implementation}/ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_1024_output_error.txt
	
module load cuda/11.3

nvcc -I./include -D TIMESTEPS=1 -D TIMESTEP_DURATION_FS=2.5e-13 -D CELL_CUTOFF_RADIUS_ANGST=30 -D CELL_LENGTH_X=10 -D CELL_LENGTH_Y=10 -D CELL_LENGTH_Z=10 -D TIME_RUN ${PROJ_DIR}/src/pdb_importer.c ${PROJ_DIR}/src/${implementation}.cu -o ${PROJ_DIR}/build/${implementation}
${PROJ_DIR}/build/${implementation} ${PROJ_DIR}/input/random_particles-1024.pdb ${PROJ_DIR}/output/${implementation}/ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_1024_output.csv
EOF

	for particle_count in 4096 16384 65536; do
		cat > ${SCRIPTS_DIR}/${implementation}_ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_${particle_count}.sh <<EOF
#!/bin/bash -l

#$ -N ${implementation}_n_${particle_count}  # Job Name
#$ -l gpus=1
#$ -l gpu_type=A40 
#$ -M eth@bu.edu
#$ -M ajamias@bu.edu
#$ -o ${PROJ_DIR}/output/${implementation}/ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_${particle_count}_output.txt
#$ -e ${PROJ_DIR}/output/${implementation}/ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_${particle_count}_output_error.txt
	
module load cuda/11.3

nvcc -I./include -D TIMESTEPS=1 -D TIMESTEP_DURATION_FS=2.5e-13 -D CELL_CUTOFF_RADIUS_ANGST=100 -D CELL_LENGTH_X=10 -D CELL_LENGTH_Y=10 -D CELL_LENGTH_Z=10 -D TIME_RUN ${PROJ_DIR}/src/pdb_importer.c ${PROJ_DIR}/src/${implementation}.cu -o ${PROJ_DIR}/build/${implementation}
${PROJ_DIR}/build/${implementation} ${PROJ_DIR}/input/random_particles-${particle_count}.pdb ${PROJ_DIR}/output/${implementation}/ts_${TIMESTEPS}_tsd_${TIMESTEP_DURATION}_n_${particle_count}_output.csv
EOF

	done
done
