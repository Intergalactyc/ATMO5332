"""
Create one WRF run directory per (event, PBL scheme): OUTDIR/<event>_<scheme>/, containing
    namelist.input  configs/namelist.input.template with the run dates and PBL scheme set
    tslist          the tower plus the mesonet stations within BOUNDS
    iofields.txt    copied from configs/
and, if given,
    --wrf-run DIR   links to the WRF run directory's executables and tables
    --met-em DIR    links to the event's met_em files ("{event}" in DIR is replaced by the event date)

Usage examples:
    python prepare_runs.py ./runs                                  (all events, core schemes)
    python prepare_runs.py ./testruns --events 20131210 --schemes ysu --run-hours 3
    python prepare_runs.py ./runs --wrf-run /lustre/work/USER/WRFV4/run --met-em ./wps/{event}

--force updates existing run directories, replacing only the files and links made here.
"""

import argparse
import csv
import os
import pathlib
import re
import shutil
import sys
from datetime import datetime, timedelta

ROOT = pathlib.Path(__file__).parent
TEMPLATE = ROOT / "configs" / "namelist.input.template"
IOFIELDS = ROOT / "configs" / "iofields.txt"
STATIONS = ROOT / "meta" / "wtm_2013_station_locations.csv"

# Event ID: UTC date of the LLJ morning
EVENTS = [
    "20131210",  # key
    "20140125",  # key
    "20131216",  # backup
    "20140122",  # backup
    "20140214",  # backup
]
INIT_OFFSET = timedelta(hours=-6)  # runs start 18 UTC the day before the event date

# bl_pbl_physics, sf_sfclay_physics (1 wherever the scheme allows it), and any other settings needed as
# (group, key, value, per_domain)
SCHEMES = {
    "ysu":      {"pbl": 1,  "sfclay": 1},
    "mynn":     {"pbl": 5,  "sfclay": 1, "extra": [("physics", "bl_mynn_closure", "2.5", False)]},
    "acm2":     {"pbl": 7,  "sfclay": 1},
    "mynn26":   {"pbl": 5,  "sfclay": 1, "extra": [("physics", "bl_mynn_closure", "2.6", False)]},
    "mynn3":    {"pbl": 5,  "sfclay": 1, "extra": [("physics", "bl_mynn_closure", "3.0", False)]},
    "boulac":   {"pbl": 8,  "sfclay": 1},
    "uw":       {"pbl": 9,  "sfclay": 1},
    "shinhong": {"pbl": 11, "sfclay": 1},
    "gbm":      {"pbl": 12, "sfclay": 1},
    "eeps":     {"pbl": 16, "sfclay": 1},
    "keps":     {"pbl": 17, "sfclay": 1},
    "mrf":      {"pbl": 99, "sfclay": 1},
    "sms3dtke": {"pbl": 0,  "sfclay": 1, "extra": [("dynamics", "km_opt", 5, True), ("dynamics", "diff_opt", 2, True)]},
    "myj":      {"pbl": 2,  "sfclay": 2},
    "gfs":      {"pbl": 3,  "sfclay": 3},
    "qnse":     {"pbl": 4,  "sfclay": 4},
    "temf":     {"pbl": 10, "sfclay": 10},
}
CORE_SCHEMES = ["ysu", "mynn", "acm2"]

TOWER = ("RTC 200m Tower", "tower", 33.610, -102.050)  # tslist (name, prefix, lat, lon)
BOUNDS = [32.10, 35.10, -103.67, -100.11]  # analysis ROI: lat min, lat max, lon min, lon max

# WRF (share/wrf_tsin.F) needs "LAT" and "LON" in the second header line at FMT='(32X,A7,2X,A7)', and reads
# each location with FMT='(A25,1X,A5,1X,F7.3,1X,F8.3)'
TSLIST_HEADER = [
    "#-----------------------------------------------#",
    "# 24 characters for name | pfx |  LAT  |   LON  |",
    "#-----------------------------------------------#",
]

# WRF run directory files not to link: per-run inputs made here, and output of earlier runs
WRF_RUN_SKIP = re.compile(r"namelist\..*|tslist|iofields.*|met_em\..*|wrf(input|bdy|out|rst)_.*|rsl\..*")


def _find(lines, group, key):
    """Line index of key in &group (None if missing) and of the group's closing '/'."""
    start = [line.strip().lower() for line in lines].index(f"&{group}")
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "/")
    return next((i for i in range(start + 1, end) if re.match(rf"\s*{key}\s*=", lines[i])), None), end


def get_int(lines, group, key, default=None):
    """First (domain 1) value of key in &group."""
    i, _ = _find(lines, group, key)
    return default if i is None else int(lines[i].split("=")[1].split(",")[0])


def set_value(lines, group, key, value):
    """Set key = value in &group, keeping the line's alignment, or add it at the end of the group."""
    i, end = _find(lines, group, key)
    if i is None:
        lines.insert(end, f" {key:<25}= {value},")
    else:
        lines[i] = re.match(r".*?=\s*", lines[i]).group() + f"{value},"


def per_domain(value, max_dom):
    return ", ".join([str(value)] * max_dom)


def make_namelist(template, event, scheme, run_hours=None):
    """namelist.input lines for one run, plus its start and end times."""
    lines = list(template)
    max_dom = get_int(lines, "domains", "max_dom")
    if run_hours:
        set_value(lines, "time_control", "run_hours", run_hours)
    start = datetime.strptime(event, "%Y%m%d") + INIT_OFFSET
    end = start + timedelta(hours=get_int(lines, "time_control", "run_hours"))
    for prefix, time in (("start", start), ("end", end)):
        for unit, fmt in (("year", "%Y"), ("month", "%m"), ("day", "%d"), ("hour", "%H")):
            set_value(lines, "time_control", f"{prefix}_{unit}", per_domain(time.strftime(fmt), max_dom))

    config = SCHEMES[scheme]
    set_value(lines, "physics", "bl_pbl_physics", per_domain(config["pbl"], max_dom))
    set_value(lines, "physics", "sf_sfclay_physics", per_domain(config["sfclay"], max_dom))
    for group, key, value, is_per_domain in config.get("extra", []):
        set_value(lines, group, key, per_domain(value, max_dom) if is_per_domain else value)
    return lines, start, end


def read_stations():
    """(archive_name, prefix, lat, lon) of each station in STATIONS. The prefix is the name's first 5 letters, or its
    first 4 and last where 5 are ambiguous (AMARN/AMARE)."""
    with open(STATIONS, newline="") as f:
        rows = list(csv.DictReader(f))
    words = [re.sub(r"[^A-Za-z0-9]", "", row["archive_name"]).upper() for row in rows]
    pfxs = [w[:5] if sum(x[:5] == w[:5] for x in words) == 1 else w[:4] + w[-1] for w in words]
    return [(row["archive_name"], pfx, float(row["latitude"]), float(row["longitude"])) for row, pfx in zip(rows, pfxs)]


def select_stations():
    """read_stations() within BOUNDS: the mesonet stations in the tslist."""
    lat_min, lat_max, lon_min, lon_max = BOUNDS
    return [s for s in read_stations() if lat_min <= s[2] <= lat_max and lon_min <= s[3] <= lon_max]


def make_tslist(locations):
    pfxs = [pfx for _, pfx, _, _ in locations]
    duplicates = sorted({pfx for pfx in pfxs if pfxs.count(pfx) > 1})
    if duplicates:
        sys.exit(f"error: tslist prefixes used more than once: {duplicates}")
    return TSLIST_HEADER + [f"{name[:25]:<25} {pfx:<5} {lat:7.3f} {lon:8.3f}" for name, pfx, lat, lon in locations]


def write_lines(path, lines):
    with open(path, "w", newline="\n") as f:  # LF even on Windows
        f.write("\n".join(lines) + "\n")


def link(source, path):
    """Symlink path -> source, replacing an existing file or link at path."""
    if path.is_symlink() or path.exists():
        path.unlink()
    os.symlink(source.resolve(), path)


def event_date(text):
    datetime.strptime(text, "%Y%m%d")  # argparse reports the ValueError
    return text


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outdir", type=pathlib.Path, help="directory to create the <event>_<scheme> run directories in")
    parser.add_argument("--events", nargs="+", type=event_date, default=EVENTS, metavar="YYYYMMDD",
                        help=f"event (morning) dates (default: all {len(EVENTS)})")
    parser.add_argument("--schemes", nargs="+", default=CORE_SCHEMES, choices=sorted(SCHEMES),
                        help=f"PBL schemes (default: {' '.join(CORE_SCHEMES)})")
    parser.add_argument("--run-hours", type=int, help="override the template run length, e.g. 3 for a test run")
    parser.add_argument("--wrf-run", type=pathlib.Path, help="WRF run directory to symlink executables and tables from")
    parser.add_argument("--met-em", metavar="DIR", help="directory of each event's met_em files to symlink; "
                                                        "{event} is replaced by the event date")
    parser.add_argument("--force", action="store_true", help="update existing run directories")
    return parser.parse_args()


def main():
    args = parse_args()
    template = TEMPLATE.read_text().splitlines()

    stations = select_stations()
    tslist = make_tslist([TOWER] + stations)
    # WRF silently ignores locations beyond max_ts_locs (default 5)
    max_ts_locs = get_int(template, "domains", "max_ts_locs", 5)
    if len(stations) + 1 > max_ts_locs:
        sys.exit(f"error: tower + {len(stations)} stations but max_ts_locs = {max_ts_locs} in {TEMPLATE.name}")

    links = {event: [] for event in args.events}
    for event in args.events:
        if args.wrf_run:
            links[event] += sorted(p for p in args.wrf_run.iterdir() if p.is_file() and not WRF_RUN_SKIP.fullmatch(p.name))
        if args.met_em:
            met_em_dir = pathlib.Path(args.met_em.replace("{event}", event))
            met_em = sorted(met_em_dir.glob("met_em.d*"))
            if not met_em:
                sys.exit(f"error: no met_em files in {met_em_dir}")
            links[event] += met_em

    runs = [(event, scheme, args.outdir / f"{event}_{scheme}") for event in args.events for scheme in args.schemes]
    for _, _, run_dir in runs:
        if run_dir.exists() and not args.force:
            sys.exit(f"error: {run_dir} exists (use --force to update it)")

    print(f"tslist: tower + {len(stations)} stations: {' '.join(s[1] for s in stations)}")
    for event, scheme, run_dir in runs:
        namelist, start, end = make_namelist(template, event, scheme, args.run_hours)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_lines(run_dir / "namelist.input", namelist)
        write_lines(run_dir / "tslist", tslist)
        shutil.copyfile(IOFIELDS, run_dir / IOFIELDS.name)
        for source in links[event]:
            link(source, run_dir / source.name)
        print(f"{run_dir}: {start:%Y-%m-%d_%H} -> {end:%Y-%m-%d_%H}, bl_pbl_physics={SCHEMES[scheme]['pbl']}, "
              f"sf_sfclay_physics={SCHEMES[scheme]['sfclay']}, {len(links[event])} links")


if __name__ == "__main__":
    main()
