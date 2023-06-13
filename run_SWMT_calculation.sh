#!/bin/bash

## loop over count, submit job to gadi with count that gets communicated to python

for y in {1991..2009}; do
   qsub -v year=$y,expt='panan-01-test-oldparams',expt_name='panan_01deg_jra55_ryf' SWMT_calculation.sh
done