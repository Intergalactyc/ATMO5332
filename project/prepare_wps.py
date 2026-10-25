"""
Create and prepare the WPS directories: OUTDIR/geo/ for geogrid (run once) and OUTDIR/<event>/ for ungrib + metgrid

OUTDIR/geo/ gets namelist.wps (configs/namelist.wps.template) and links to geogrid.exe and the geogrid/ table directory.
Each OUTDIR/<event>/ gets:
    namelist.wps    the template with the event's dates (d01 for the whole run, nests at the start only) and the
                    geo_em files read from ../geo/
    links           ungrib.exe, metgrid.exe, the metgrid/ table directory, Vtable (Vtable.ECMWF), and
                    GRIBFILE.AAA, GRIBFILE.AAB, ... to the event's ERA5 GRIB files (ERA5_DIR/<event>/*.grib from era5.py)
The run length is taken from configs/namelist.input.template, so WPS covers the same period as the WRF runs.

Usage examples:
    python prepare_wps.py ./wps --wps /lustre/work/USER/WPSV4                  (all events)
    python prepare_wps.py ./wps --wps /lustre/work/USER/WPSV4 --events 20131210 --era5 /lustre/work/USER/era5
    python prepare_wps.py ./wps --wps /lustre/work/USER/WPSV4 --geog-data-path /lustre/work/USER/WPS_GEOG

Existing directories are only updated with --force, which replaces the files and links made here and leaves
anything else (e.g. geo_em or met_em output) alone.
"""

import argparse
import pathlib
import sys
from datetime import datetime, timedelta

from prepare_runs import EVENTS, INIT_OFFSET, ROOT, TEMPLATE, event_date, get_int, link, set_value, write_lines

WPS_TEMPLATE = ROOT / "configs" / "namelist.wps.template"
ERA5 = ROOT / "era5"
VTABLE = "ungrib/Variable_Tables/Vtable.ECMWF"


def gribfile(i):
    """Name link_grib.csh gives the i-th GRIB file: GRIBFILE.AAA, GRIBFILE.AAB, ..."""
    return "GRIBFILE." + "".join(chr(ord("A") + i // 26 ** k % 26) for k in (2, 1, 0))


def make_namelist_wps(template, event, run_hours):
    """namelist.wps lines for one event, plus its start and end times."""
    lines = list(template)
    max_dom = get_int(lines, "share", "max_dom")
    start = datetime.strptime(event, "%Y%m%d") + INIT_OFFSET
    end = start + timedelta(hours=run_hours)
    dates = lambda times: ", ".join(f"'{t:%Y-%m-%d_%H:%M:%S}'" for t in times)
    set_value(lines, "share", "start_date", dates([start] * max_dom))
    set_value(lines, "share", "end_date", dates([end] + [start] * (max_dom - 1)))  # nests only need the initial time
    set_value(lines, "share", "opt_output_from_geogrid_path", "'../geo/'")
    return lines, start, end


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outdir", type=pathlib.Path, help="directory to create the geo/ and <event>/ directories in")
    parser.add_argument("--wps", type=pathlib.Path, required=True,
                        help="WPS directory to link executables and tables from (e.g. /lustre/work/USER/WPSV4)")
    parser.add_argument("--era5", type=pathlib.Path, default=ERA5,
                        help="directory with the <event>/*.grib files from era5.py (default: %(default)s)")
    parser.add_argument("--events", nargs="+", type=event_date, default=EVENTS, metavar="YYYYMMDD",
                        help=f"event (morning) dates (default: all {len(EVENTS)})")
    parser.add_argument("--geog-data-path", metavar="DIR",
                        help="static geographic data directory for geogrid (default: geog_data_path in the template)")
    parser.add_argument("--force", action="store_true", help="update existing directories")
    return parser.parse_args()


def main():
    args = parse_args()
    template = WPS_TEMPLATE.read_text().splitlines()
    if args.geog_data_path:
        set_value(template, "geogrid", "geog_data_path", f"'{args.geog_data_path}'")
    run_hours = get_int(TEMPLATE.read_text().splitlines(), "time_control", "run_hours")

    missing = [name for name in ("geogrid.exe", "ungrib.exe", "metgrid.exe", "geogrid", "metgrid", VTABLE)
               if not (args.wps / name).exists()]
    if missing:
        sys.exit(f"error: {args.wps} has no {', '.join(missing)}")
    gribs = {event: sorted((args.era5 / event).glob("*.grib")) for event in args.events}
    for event, files in gribs.items():
        if not files:
            sys.exit(f"error: no ERA5 GRIB files in {args.era5 / event}")

    geo_dir = args.outdir / "geo"
    for d in [geo_dir] + [args.outdir / event for event in args.events]:
        if d.exists() and not args.force:
            sys.exit(f"error: {d} exists (use --force to update it)")

    geo_dir.mkdir(parents=True, exist_ok=True)
    write_lines(geo_dir / "namelist.wps", template)
    for name in ("geogrid.exe", "geogrid"):
        link(args.wps / name, geo_dir / name)
    print(f"{geo_dir}: geogrid")

    for event in args.events:
        event_dir = args.outdir / event
        lines, start, end = make_namelist_wps(template, event, run_hours)
        event_dir.mkdir(parents=True, exist_ok=True)
        write_lines(event_dir / "namelist.wps", lines)
        for name in ("ungrib.exe", "metgrid.exe", "metgrid"):
            link(args.wps / name, event_dir / name)
        link(args.wps / VTABLE, event_dir / "Vtable")
        for old in event_dir.glob("GRIBFILE.*"):
            old.unlink()
        for i, grib in enumerate(gribs[event]):
            link(grib, event_dir / gribfile(i))
        print(f"{event_dir}: {start:%Y-%m-%d_%H} -> {end:%Y-%m-%d_%H}, {len(gribs[event])} GRIB files")


if __name__ == "__main__":
    main()
