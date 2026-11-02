"""
Pack WRF run results into one .tar.gz for transfer. For each run directory RUNS_DIR/<event>_<scheme>/:
    tower_column.nc     the d03 column at the tower's grid point from every wrfout_d03 file (written into the run
                        directory first): heights, earth-relative winds, potential temperature, pressure, moisture,
                        TKE and exchange coefficients on the lowest --levels levels, plus PBLH, UST, HFX, LH
    <pfx>.d03.TS, .UU, .VV, .WW, .TH, .QV, .PH, .PR    tower and station time series (d03 only)
    namelist.input, namelist.output, tslist
    rsl.error.0000, real_logs/rsl.*.0000, wrf*-<jobid>.log   logs (timing, level heights, errors)
Files are stored as <event>_<scheme>/<file> in the archive. The wrfout files themselves are not packed.

The tower's grid point is the one WRF used for its time series (first line of tower.d03.TS), so the column and the
station files line up. Reading the wrfout files takes most of the time; --workers reads several at once (run it as
a batch job or on an interactive compute node then, not a login node).

Usage examples:
    python pack_results.py ./runs --workers 16                       (all runs -> results.tar.gz)
    python pack_results.py ./runs --events 20131210 --schemes ysu -o test.tar.gz
"""

import argparse
import multiprocessing
import pathlib
import re
import sys
import tarfile
from datetime import datetime

import numpy as np
from netCDF4 import Dataset, chartostring

from prepare_runs import TOWER
from submit import select

PATTERNS = ["tower_column.nc", "*.d03.??", "namelist.input", "namelist.output", "tslist", "rsl.error.0000",
            "real_logs/rsl.error.0000", "real_logs/rsl.out.0000", "wrf*-*.log"]
G = 9.81  # m s-2, as in WRF's own time series heights
UNITS = {"z": "m above ground (mass levels)", "z_w": "m above ground (w levels)",
         "u": "m s-1 (earth-relative)", "v": "m s-1 (earth-relative)", "theta": "K (dry potential temperature)",
         "p": "Pa", "qv": "kg kg-1", "tke_pbl": "m2 s-2", "exch_h": "m2 s-1", "exch_m": "m2 s-1",
         "pblh": "m", "ust": "m s-1", "hfx": "W m-2", "lh": "W m-2"}


def tower_point(run):
    """0-based (i, j) of the mass grid point WRF used for the tower, from the first line of tower.d03.TS."""
    header = (run / f"{TOWER[1]}.d03.TS").open().readline()
    i, j = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", header).groups()  # "( ix, iy)", 1-based
    return int(i) - 1, int(j) - 1


def read_column(task):
    """Column values at mass point (i, j) from one wrfout file, on the lowest `levels` levels (+1 for w levels)."""
    path, i, j, levels = task
    with Dataset(path) as nc:
        nc.set_auto_mask(False)
        v = nc.variables
        ph = v["PH"][0, :levels + 1, j, i] + v["PHB"][0, :levels + 1, j, i]  # geopotential at the w levels
        hgt = v["HGT"][0, j, i]
        u = 0.5 * (v["U"][0, :levels, j, i] + v["U"][0, :levels, j, i + 1])  # unstagger to the mass point
        vv = 0.5 * (v["V"][0, :levels, j, i] + v["V"][0, :levels, j + 1, i])
        cosa, sina = v["COSALPHA"][0, j, i], v["SINALPHA"][0, j, i]
        column = {"time": str(chartostring(v["Times"][0])),
                  "z": 0.5 * (ph[:-1] + ph[1:]) / G - hgt,
                  "z_w": ph / G - hgt,
                  "u": u * cosa - vv * sina,  # same rotation WRF uses for its time series
                  "v": vv * cosa + u * sina,
                  "theta": v["T"][0, :levels, j, i] + 300.0,
                  "p": v["P"][0, :levels, j, i] + v["PB"][0, :levels, j, i],
                  "qv": v["QVAPOR"][0, :levels, j, i]}
        for name in ("TKE_PBL", "EXCH_H", "EXCH_M"):  # on the w levels; TKE_PBL only exists for TKE schemes
            if name in v:
                column[name.lower()] = v[name][0, :levels + 1, j, i]
        for name in ("PBLH", "UST", "HFX", "LH"):
            column[name.lower()] = v[name][0, j, i]
    return column


def write_column(run, i, j, levels, columns):
    columns.sort(key=lambda c: c["time"])
    times = [datetime.strptime(c["time"], "%Y-%m-%d_%H:%M:%S") for c in columns]
    with Dataset(run / "tower_column.nc", "w") as nc:
        nc.title = f"WRF d03 column at the tower grid point (i={i + 1}, j={j + 1}, 1-based), run {run.name}"
        nc.createDimension("time", len(columns))
        nc.createDimension("level", levels)
        nc.createDimension("level_w", levels + 1)
        time = nc.createVariable("time", "f8", ("time",))
        time.units = f"minutes since {times[0]:%Y-%m-%d %H:%M:%S}"
        time[:] = [(t - times[0]).total_seconds() / 60 for t in times]
        for name in columns[0]:
            if name == "time":
                continue
            data = np.array([c[name] for c in columns])
            dims = ("time",) if data.ndim == 1 else ("time", "level" if data.shape[1] == levels else "level_w")
            var = nc.createVariable(name, "f4", dims)
            var.units = UNITS[name]
            var[:] = data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", type=pathlib.Path, help="runs directory made by prepare_runs.py")
    parser.add_argument("--events", nargs="+", metavar="YYYYMMDD", help="only these events (default: all)")
    parser.add_argument("--schemes", nargs="+", help="only these PBL schemes (default: all)")
    parser.add_argument("--levels", type=int, default=30, help="model levels in the tower column (default: 30)")
    parser.add_argument("--workers", type=int, default=1, help="wrfout files to read at once (default: 1)")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("results.tar.gz"),
                        help="archive to write (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        sys.exit(f"error: {args.output} exists")
    runs = select(args.dir, args.events, args.schemes)

    # Progress lines are flushed right away so they show up in the Slurm log while the job runs
    with multiprocessing.Pool(args.workers) as pool:
        for run in runs:
            rsl = run / "rsl.error.0000"
            if not (rsl.is_file() and "SUCCESS COMPLETE WRF" in rsl.read_text(errors="replace")):
                print(f"warning: {run.name} has not finished (no SUCCESS COMPLETE WRF in rsl.error.0000)", flush=True)
            wrfouts = sorted(run.glob("wrfout_d03_*"))
            if not wrfouts:
                print(f"warning: {run.name} has no wrfout_d03 files; no tower column", flush=True)
                continue
            i, j = tower_point(run)
            columns = pool.map(read_column, [(f, i, j, args.levels) for f in wrfouts])
            write_column(run, i, j, args.levels, columns)
            print(f"{run.name}: tower column from {len(columns)} wrfout files", flush=True)

    total = count = 0
    with tarfile.open(args.output, "w:gz", compresslevel=6) as tar:
        for run in runs:
            files = sorted({f for pattern in PATTERNS for f in run.glob(pattern) if f.is_file()})
            for f in files:
                tar.add(f, arcname=f"{run.name}/{f.relative_to(run).as_posix()}")
            size = sum(f.stat().st_size for f in files)
            total, count = total + size, count + len(files)
            print(f"{run.name}: packed {len(files)} files ({size / 1e6:.1f} MB)", flush=True)
    print(f"wrote {args.output}: {count} files from {len(runs)} runs, "
          f"{total / 1e6:.1f} MB -> {args.output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
