"""
Write a reduced copy of each run's wrfout_d03 files for transfer: only VARIABLES, and only the lowest --levels model
levels (about 2.5 km for the default 35). Output goes to OUTDIR/<event>_<scheme>/wrfout_d03_<time>, keeping WRF's
global attributes, so the files can still be read like ordinary wrfout files. The ':' in the times becomes '_'
(e.g. wrfout_d03_2013-12-10_00_00_00), since Windows doesn't allow ':' in file names.

Files that already exist in OUTDIR are skipped, so an interrupted run can be restarted. Use --workers to process
several files at once; run it as a batch job rather than on a login node, e.g.:
    sbatch -p nocona -N 1 --ntasks-per-node=32 -t 02:00:00 -o subset-%j.log \
        --wrap "python3 subset_wrfout.py ./runs ./wrfout_subset --workers 32"

Usage examples:
    python3 subset_wrfout.py ./runs ./wrfout_subset --workers 32
    python3 subset_wrfout.py ./runs ./wrfout_subset --events 20131210 --schemes ysu    (one run, one process)

Then copy OUTDIR to your computer with rsync. Requires netCDF4 and tqdm.
"""

import argparse
import multiprocessing
import os
import pathlib
import sys

from netCDF4 import Dataset
from tqdm import tqdm

from submit import select

VARIABLES = ["Times", "XLAT", "XLONG", "HGT",
             "U", "V", "W", "T", "P", "PB", "PH", "PHB", "QVAPOR", "TKE_PBL", "EXCH_H", "EXCH_M",
             "PBLH", "UST", "HFX", "LH", "T2", "Q2", "U10", "V10", "PSFC", "TSK"]


def subset_file(task):
    """Write the reduced copy of one wrfout file; returns the requested variables it doesn't have."""
    source, target, levels = task
    sizes = {"bottom_top": levels, "bottom_top_stag": levels + 1}  # staggered levels bound the kept layers
    partial = target.with_name(target.name + ".part")  # renamed when complete, so restarts skip only whole files
    with Dataset(source) as src, Dataset(partial, "w", format="NETCDF4") as dst:
        src.set_auto_mask(False)
        dst.setncatts({name: src.getncattr(name) for name in src.ncattrs()})
        if "BOTTOM-TOP_GRID_DIMENSION" in dst.ncattrs():
            dst.setncattr("BOTTOM-TOP_GRID_DIMENSION", levels + 1)
        for name, dim in src.dimensions.items():
            dst.createDimension(name, None if dim.isunlimited() else min(sizes.get(name, len(dim)), len(dim)))
        missing = [name for name in VARIABLES if name not in src.variables]
        for name in VARIABLES:
            if name in missing:
                continue
            var = src.variables[name]
            out = dst.createVariable(name, var.dtype, var.dimensions, zlib=var.dtype.kind == "f", complevel=2,
                                     shuffle=True)
            out.setncatts({attr: var.getncattr(attr) for attr in var.ncattrs() if attr != "_FillValue"})
            out[:] = var[tuple(slice(0, sizes[d]) if d in sizes else slice(None) for d in var.dimensions)]
    os.replace(partial, target)
    return missing


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", type=pathlib.Path, help="runs directory made by prepare_runs.py")
    parser.add_argument("outdir", type=pathlib.Path, help="directory to write the reduced files to")
    parser.add_argument("--events", nargs="+", metavar="YYYYMMDD", help="only these events (default: all)")
    parser.add_argument("--schemes", nargs="+", help="only these PBL schemes (default: all)")
    parser.add_argument("--levels", type=int, default=35, help="number of lowest model levels to keep (default: 35)")
    parser.add_argument("--workers", type=int, default=1, help="files to process at once (default: 1)")
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = []
    for run in select(args.dir, args.events, args.schemes):
        (args.outdir / run.name).mkdir(parents=True, exist_ok=True)
        for source in sorted(run.glob("wrfout_d03_*")):
            target = args.outdir / run.name / source.name.replace(":", "_")
            if not target.exists():
                tasks.append((source, target, args.levels))
    if not tasks:
        sys.exit("nothing to do: no wrfout_d03 files without an existing reduced copy")

    missing = set()
    with multiprocessing.Pool(args.workers) as pool:
        for result in tqdm(pool.imap_unordered(subset_file, tasks), total=len(tasks), unit="file", desc="subsetting"):
            missing.update(result)
    if missing:
        print(f"note: not in every wrfout file (left out where absent): {' '.join(sorted(missing))}")
    print(f"wrote {len(tasks)} files to {args.outdir}")


if __name__ == "__main__":
    main()
