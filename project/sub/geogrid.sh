#!/bin/bash
#SBATCH -J geogrid
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -p nocona
#SBATCH -N 1
#SBATCH --ntasks-per-node=30
#SBATCH -t 00:30:00

# Usage: sbatch geogrid.sh WPS_DIR/geo        (directory made by prepare_wps.py)

cd "${1:?usage: sbatch geogrid.sh WPS_DIR/geo}" || exit 1

source /lustre/work/bancell/spack/share/spack/setup-env.sh
spack load wps

mpirun -np 30 ./geogrid.exe

if ! grep -q "Successful completion of geogrid" geogrid.log*; then
    echo "geogrid failed; see $PWD/geogrid.log*"
    exit 1
fi
