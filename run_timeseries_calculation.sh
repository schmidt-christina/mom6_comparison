#!/bin/bash

## loop over count, submit job to gadi with count that gets communicated to python

for y in {1991..2009..10}; do
   qsub -v year=$y,expt='panant-01-zstar-ACCESSyr2',expt_name='panan_01deg_jra55_ryf' Timeseries_calculation.sh
done

for y in {1991..2010}; do
    qsub -v year=$y,expt='panant-005-zstar-ACCESSyr2',expt_name='panan_005deg_jra55_ryf' Timeseries_calculation.sh
done
