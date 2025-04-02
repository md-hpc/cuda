#!/bin/bash

cd output
for i in nsquared nsquared_shared nsquared_n3l; do  
cd ${i}	
for j in 1024 4096 16384 65536; do
	    cat ts-1_tsd-1e-15_n-${j}_output.txt
done
cd ..
done

