#!/bin/bash

## loop over count, submit job to gadi with count that gets communicated to python

# for y in {1991..2002}; do
#    qsub -v year=$y,expt='panant-01-zstar-ACCESSyr2',expt_name='panan_01deg_jra55_ryf',contour_depth=3000 Shelf_export_calculation.sh
#    qsub -v year=$y,expt='panant-01-zstar-ACCESSyr2',expt_name='panan_01deg_jra55_ryf',contour_depth=3500 Shelf_export_calculation.sh
#    qsub -v year=$y,expt='panant-01-zstar-ACCESSyr2',expt_name='panan_01deg_jra55_ryf',contour_depth=4000 Shelf_export_calculation.sh
# done

# for y in {1991..2002}; do
#     qsub -v year=$y,expt='panant-005-zstar-ACCESSyr2',expt_name='panan_005deg_jra55_ryf',contour_depth=3000 Shelf_export_calculation.sh
#     qsub -v year=$y,expt='panant-005-zstar-ACCESSyr2',expt_name='panan_005deg_jra55_ryf',contour_depth=3500 Shelf_export_calculation.sh
#     qsub -v year=$y,expt='panant-005-zstar-ACCESSyr2',expt_name='panan_005deg_jra55_ryf',contour_depth=4000 Shelf_export_calculation.sh
# done

for y in {1991..1999}; do
    qsub -v year=$y,expt='panant-0025-zstar-ACCESSyr2',expt_name='panan_0025deg_jra55_ryf',contour_depth=1000 Shelf_export_calculation.sh
    qsub -v year=$y,expt='panant-0025-zstar-ACCESSyr2',expt_name='panan_0025deg_jra55_ryf',contour_depth=2500 Shelf_export_calculation.sh
done