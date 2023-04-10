#!/bin/bash

## loop over count, submit job to gadi with count that gets communicated to python

for y in {1991..2019}; do
   qsub -v year=$y,expt='global-01-v3' SWMT_calculation.sh
done