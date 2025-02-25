#!/bin/bash

qsub nsquared_ts-1_tsd-1e-15.sh 1024
qsub nsquared_ts-1_tsd-1e-15.sh 4096
qsub nsquared_ts-1_tsd-1e-15.sh 16384
qsub nsquared_ts-1_tsd-1e-15.sh 65536
# and so on...
