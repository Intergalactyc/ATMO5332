#!/bin/bash
#SBATCH -J ungrib
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -p nocona
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -t 02:00:00

# Usage: sbatch ungrib.sh WPS_DIR/EVENT       (directory made by prepare_wps.py)

cd "${1:?usage: sbatch ungrib.sh WPS_DIR/EVENT}" || exit 1

source /lustre/work/bancell/spack/share/spack/setup-env.sh
spack load wps

./ungrib.exe

if ! grep -q "Successful completion of ungrib" ungrib.log; then
    echo "ungrib failed; see $PWD/ungrib.log"
    exit 1
fi
