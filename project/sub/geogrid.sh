#!/bin/bash
#SBATCH -J geogrid
#SBATCH -o %x-%j.log
#SBATCH -p nocona
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -t 00:30:00

# Usage: sbatch -D WPS_DIR/geo geogrid.sh     (runs in the directory made by prepare_wps.py; see submit.py)
# Slurm's output (stdout and stderr) goes to geogrid-<jobid>.log in that directory.

rm -f geogrid.log*  # logs from an earlier run

source /lustre/work/bancell/spack/share/spack/setup-env.sh
spack load wps

mpirun -np 32 ./geogrid.exe

if ! grep -q "Successful completion of geogrid" geogrid.log*; then
    echo "geogrid failed; see $PWD/geogrid.log*"
    exit 1
fi
