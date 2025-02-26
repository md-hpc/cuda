#!/bin/bash

for i in 1024 4096 16384 65536
    do
	qsub scripts/nsquared_ts-1_tsd-1e-15_n-${i}.sh 
    done
