#!/bin/bash
#PBS -P e14
#PBS -l ncpus=48
#PBS -l mem=180GB
#PBS -q normal
#PBS -l walltime=0:40:00
#PBS -l storage="gdata/hh5+gdata/ik11+gdata/v45+gdata/e14+gdata/cj50+scratch/v45+scratch/x77"
#PBS -l wd
#PBS -o calculation_Shelf_export.out
#PBS -j oe

module use /g/data/hh5/public/modules

python3 Shelf_export_calculation.py ${year} ${expt} ${expt_name} ${contour_depth} &>> Shelf_export_calculation_${expt}_${year}_${contour_depth}.txt