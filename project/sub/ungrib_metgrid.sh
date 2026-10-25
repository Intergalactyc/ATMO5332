#!/bin/bash
#SBATCH -J ungrib_metgrid
#SBATCH -o %x-%j.log
#SBATCH -p nocona
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH -t 02:00:00

# Usage: sbatch -D WPS_DIR/EVENT ungrib_metgrid.sh     (after geogrid has finished; see submit.py)
# Runs ungrib.exe, then metgrid.exe if ungrib succeeded. Slurm's output goes to <jobname>-<jobid>.log there.

rm -f ungrib.log metgrid.log*  # logs from an earlier run

source /lustre/work/bancell/spack/share/spack/setup-env.sh
spack load wps

./ungrib.exe

if ! grep -q "Successful completion of ungrib" ungrib.log; then
    echo "ungrib failed; see $PWD/ungrib.log"
    exit 1
fi

mpirun -np 48 ./metgrid.exe

if ! grep -q "Successful completion of metgrid" metgrid.log*; then
    echo "metgrid failed; see $PWD/metgrid.log*"
    exit 1
fi
