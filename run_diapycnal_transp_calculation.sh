#!/bin/bash

## loop over count, submit job to gadi with count that gets communicated to python

m=1  # just a dummy variable
for y in {2003..2005}; do
   qsub -v year=$y,month=$m,expt='panant-01-zstar-ACCESSyr2',expt_name='panan_01deg_jra55_ryf' Diapycnal_transport_calculation.sh
done


for y in {2003..2010}; do
    for m in {1..12}; do
        qsub -v year=$y,month=$m,expt='panant-005-zstar-ACCESSyr2',expt_name='panan_005deg_jra55_ryf' Diapycnal_transport_calculation.sh
    done
done

