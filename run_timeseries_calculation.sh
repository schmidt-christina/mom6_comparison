#!/bin/bash

## loop over count, submit job to gadi with count that gets communicated to python

for y in {1991..2051..10}; do
   qsub -v year=$y,expt='panant-01-zstar-v13' Timeseries_calculation.sh
done

for y in {1991..1991..10}; do
   qsub -v year=$y,expt='panant-01-hycom1-v13' Timeseries_calculation.sh
done

for y in {1991..2011..10}; do
   qsub -v year=$y,expt='panant-01-zstar-ACCESSyr2' Timeseries_calculation.sh
done

for y in {1991..2021..10}; do
   qsub -v year=$y,expt='global-01-v1' Timeseries_calculation.sh
done

for y in {1991..2011..10}; do
   qsub -v year=$y,expt='global-01-v2' Timeseries_calculation.sh
done

for y in {1991..2011..10}; do
   qsub -v year=$y,expt='global-01-v3' Timeseries_calculation.sh
done

for y in {1991..1997}; do
   qsub -v year=$y,expt='panan_005deg_jra55_ryf' Timeseries_calculation.sh
done