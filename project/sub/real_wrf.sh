#!/bin/bash
#SBATCH -J wrf
#SBATCH -o %x-%j.log
#SBATCH -p nocona
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 24:00:00

# Usage: sbatch -D RUNS_DIR/EVENT_SCHEME real_wrf.sh     (runs in the directory made by prepare_runs.py; see submit.py)
# Runs real.exe, then wrf.exe if real succeeded; real's rsl logs are moved to real_logs/.
# Slurm's output goes to <jobname>-<jobid>.log there.

rm -rf rsl.* real_logs  # logs from an earlier run

module load gcc/14.2.0 openmpi/5.0.5 wrf/4.5.2-mpi netcdf-fortran/4.6.1-mpi netcdf-c/4.9.2-mpi

mpirun -np 128 ./real.exe

if ! grep -q "SUCCESS COMPLETE REAL_EM INIT" rsl.error.0000; then
    echo "real.exe failed; see $PWD/rsl.error.0000"
    exit 1
fi
mkdir -p real_logs
mv rsl.* real_logs/

mpirun -np 128 ./wrf.exe

if ! grep -q "SUCCESS COMPLETE WRF" rsl.error.0000; then
    echo "wrf.exe failed; see $PWD/rsl.error.0000"
    exit 1
fi
