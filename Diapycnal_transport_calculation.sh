#!/bin/bash
#PBS -P e14
#PBS -l ncpus=48
#PBS -l mem=180GB
#PBS -l jobfs=100GB
#PBS -q normal
#PBS -l walltime=5:00:00
#PBS -l storage="gdata/hh5+gdata/ik11+gdata/v45+gdata/e14+gdata/cj50+scratch/v45+scratch/x77"
#PBS -l wd
#PBS -o calculation_diapycnal_transp.out
#PBS -j oe

module use /g/data/hh5/public/modules

python3 Diapycnal_transport_calculation.py ${year} ${month} ${expt} ${expt_name} &>> Diapycnal_transport_calculation_${expt}_${year}_${month}.txt