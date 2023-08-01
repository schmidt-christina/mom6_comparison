#!/bin/bash

## loop over count, submit job to gadi with count that gets communicated to python

for y in {1998..2002}; do
   qsub -v year=$y,expt='panant-005-zstar-ACCESSyr2',expt_name='panan_005deg_jra55_ryf' SWMT_calculation.sh
done