#!/bin/bash

for particle_count in 1024 4096 16384 65536
do
for implementation in nsquared nsquared_shared nsquared_n3l
do
	qsub scripts/${implementation}_ts-1_tsd-1e-15_n-${particle_count}.sh
done
done

