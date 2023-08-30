#!/bin/bash

## loop over count, submit job to gadi with count that gets communicated to python

for y in {1991..1999}; do
   qsub -v year=$y,expt='panant-0025-zstar-ACCESSyr2',expt_name='panan_0025deg_jra55_ryf' SWMT_calculation.sh
done