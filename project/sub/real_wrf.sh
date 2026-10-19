#!/bin/bash
#SBATCH -J wrf
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -p nocona
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 24:00:00

# Usage: sbatch -J wrf_EVENT_SCHEME real_wrf.sh RUNS_DIR/EVENT_SCHEME     (directory made by prepare_runs.py)
# Runs real.exe, then wrf.exe if real succeeded. real's rsl logs are moved to real_logs/.

cd "${1:?usage: sbatch real_wrf.sh RUNS_DIR/EVENT_SCHEME}" || exit 1

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
