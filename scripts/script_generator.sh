#!/bin/bash
rm -rf nsquared*.sh
for particle_count in 1024 4096 16384 65536
do
	for implementation in nsquared nsquared_shared nsquared_n3l
	do
		cat > ${implementation}_ts-1_tsd-1e-15_n-${particle_count}.sh <<EOF
#!/bin/bash -l

#$ -N ts-1_tsd-1e-15_n-${particle_count}  # Job Name
#$ -l gpus=1
#$ -l gpu_type=A40 
#$ -M eth@bu.edu
#$ -M ajamias@bu.edu
#$ -o output/${implementation}/ts-1_tsd-1e-15_n-${particle_count}_output.txt
#$ -e output/${implementation}/ts-1_tsd-1e-15_n-${particle_count}_output_error.txt
	
module load cuda/11.3

nvcc -I./include -D TIMESTEPS=1 -D TIMESTEP_DURATION_FS=1e-15 -D CELL_CUTOFF_RADIUS_ANGST=10 -D UNIVERSE_LENGTH=100 -D TIME_RUN -D CELL_LENGTH_X=10 CELL_LENGTH_Y=10 CELL_LENGTH_Z=10 src/pdb_importer.c src/${implementation}.cu -o build/${implementation}
time build/${implementation} input/random_particles-${particle_count}.pdb output/${implementation}/ts-1_tsd-1e-15_n-${particle_count}_output.csv

EOF

	done
done
