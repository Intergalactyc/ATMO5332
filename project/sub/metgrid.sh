#!/bin/bash
#SBATCH -J metgrid
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -p nocona
#SBATCH -N 1
#SBATCH --ntasks-per-node=38
#SBATCH -t 01:00:00

# Usage: sbatch metgrid.sh WPS_DIR/EVENT      (after geogrid and this event's ungrib have finished)

cd "${1:?usage: sbatch metgrid.sh WPS_DIR/EVENT}" || exit 1

source /lustre/work/bancell/spack/share/spack/setup-env.sh
spack load wps

mpirun -np 38 ./metgrid.exe

if ! grep -q "Successful completion of metgrid" metgrid.log*; then
    echo "metgrid failed; see $PWD/metgrid.log*"
    exit 1
fi
