"""
Submit the WPS or WRF batch jobs (sub/*.sh). Each job runs in its own directory (sbatch -D), so its Slurm log
<jobname>-<jobid>.log ends up next to the program's own logs.

    wps DIR   geogrid in DIR/geo, then ungrib + metgrid in each DIR/<event> once geogrid has succeeded
    wrf DIR   real + wrf in each DIR/<event>_<scheme>

Usage examples (after prepare_wps.py / prepare_runs.py):
    python3 submit.py wps ./wps                                     (geogrid + all events)
    python3 submit.py wps ./wps --events 20131216 --skip-geogrid    (redo one event; geo_em files already made)
    python3 submit.py wrf ./runs                                    (every run directory)
    python3 submit.py wrf ./runs --events 20131210 --schemes ysu mynn
    python3 submit.py wrf ./runs --dry-run                          (print the sbatch commands only)
"""

import argparse
import pathlib
import subprocess
import sys

SUB = pathlib.Path(__file__).parent / "sub"


def sbatch(options, script, dry_run):
    """Submit script with the given sbatch options; returns the job ID."""
    command = ["sbatch", "--parsable"] + options + [str(SUB / script)]
    print(" ".join(command))
    if dry_run:
        return "JOBID"
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, universal_newlines=True)
    return result.stdout.strip().split(";")[0]  # --parsable prints "jobid" or "jobid;cluster"


def select(directory, events, schemes=None):
    """Subdirectories of directory named <event> or <event>_<scheme>, restricted to the given events and schemes."""
    selected = []
    for d in sorted(p for p in directory.iterdir() if p.is_dir() and p.name != "geo"):
        event, _, scheme = d.name.partition("_")
        if (not events or event in events) and (not schemes or scheme in schemes):
            selected.append(d.resolve())
    if not selected:
        sys.exit(f"error: no matching directories in {directory}")
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage")
    stages.required = True
    wps = stages.add_parser("wps", help="geogrid, then ungrib + metgrid for each event")
    wps.add_argument("dir", type=pathlib.Path, help="WPS directory made by prepare_wps.py")
    wps.add_argument("--events", nargs="+", metavar="YYYYMMDD", help="only these events (default: all)")
    wps.add_argument("--skip-geogrid", action="store_true", help="don't run geogrid (its output already exists)")
    wrf = stages.add_parser("wrf", help="real + wrf for each run")
    wrf.add_argument("dir", type=pathlib.Path, help="runs directory made by prepare_runs.py")
    wrf.add_argument("--events", nargs="+", metavar="YYYYMMDD", help="only these events (default: all)")
    wrf.add_argument("--schemes", nargs="+", help="only these PBL schemes (default: all)")
    for stage in (wps, wrf):
        stage.add_argument("--dry-run", action="store_true", help="print the sbatch commands without submitting")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage == "wps":
        after_geogrid = []
        if not args.skip_geogrid:
            geogrid = sbatch(["-D", str((args.dir / "geo").resolve())], "geogrid.sh", args.dry_run)
            after_geogrid = [f"--dependency=afterok:{geogrid}"]
        for d in select(args.dir, args.events):
            sbatch(["-D", str(d), "-J", f"wps_{d.name}"] + after_geogrid, "ungrib_metgrid.sh", args.dry_run)
    else:
        for d in select(args.dir, args.events, args.schemes):
            sbatch(["-D", str(d), "-J", f"wrf_{d.name}"], "real_wrf.sh", args.dry_run)


if __name__ == "__main__":
    main()
