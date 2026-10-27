"""
Pack WRF run results into one .tar.gz for transfer. For each run directory RUNS_DIR/<event>_<scheme>/:
    <pfx>.d03.TS, .UU, .VV, .WW, .TH, .QV, .PH, .PR    tower and station time series (d03 only)
    namelist.input, namelist.output, tslist
    rsl.error.0000, real_logs/rsl.*.0000, wrf*-<jobid>.log   logs (timing, level heights, errors)
    wrfout_d03_*                                            only with --wrfout (tens of GB per run)
Files are stored as <event>_<scheme>/<file> in the archive.

Usage examples:
    python3 pack_results.py ./runs                                    (all runs -> results.tar.gz)
    python3 pack_results.py ./runs --events 20131210 --schemes ysu -o test.tar.gz
    python3 pack_results.py ./runs --wrfout -o with_wrfout.tar.gz

Requires tqdm (progress bar).
"""

import argparse
import pathlib
import sys
import tarfile

from tqdm import tqdm

from submit import select

PATTERNS = ["*.d03.??", "namelist.input", "namelist.output", "tslist", "rsl.error.0000",
            "real_logs/rsl.error.0000", "real_logs/rsl.out.0000", "wrf*-*.log"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", type=pathlib.Path, help="runs directory made by prepare_runs.py")
    parser.add_argument("--events", nargs="+", metavar="YYYYMMDD", help="only these events (default: all)")
    parser.add_argument("--schemes", nargs="+", help="only these PBL schemes (default: all)")
    parser.add_argument("--wrfout", action="store_true", help="also include the wrfout_d03 files")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("results.tar.gz"),
                        help="archive to write (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        sys.exit(f"error: {args.output} exists")
    patterns = PATTERNS + (["wrfout_d03_*"] if args.wrfout else [])
    runs = select(args.dir, args.events, args.schemes)

    files = {}  # run -> [(file, size)]
    for run in runs:
        rsl = run / "rsl.error.0000"
        if not (rsl.is_file() and "SUCCESS COMPLETE WRF" in rsl.read_text(errors="replace")):
            print(f"warning: {run.name} has not finished (no SUCCESS COMPLETE WRF in rsl.error.0000)")
        found = sorted({f for pattern in patterns for f in run.glob(pattern) if f.is_file()})
        files[run] = [(f, f.stat().st_size) for f in found]
    total = sum(size for run_files in files.values() for _, size in run_files)

    # Progress is counted in bytes, so large wrfout files and small time series files are weighted fairly
    with tarfile.open(args.output, "w:gz") as tar, \
            tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="packing") as bar:
        for run, run_files in files.items():
            bar.set_postfix_str(run.name)
            for f, size in run_files:
                tar.add(f, arcname=f"{run.name}/{f.relative_to(run).as_posix()}")
                bar.update(size)
    count = sum(len(run_files) for run_files in files.values())
    print(f"wrote {args.output}: {count} files from {len(runs)} runs, "
          f"{total / 1e6:.1f} MB -> {args.output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
