"""
Generate a tslist file with the 200 meter tower and the mesonet stations inside a lat/lon bounding box.

Run this after make_namelists.py. A tslist file will be written into every run directory of the output (runs)
directory, i.e. every immediate subdirectory that contains a namelist.input. Station coordinates are taken as of
each run's start time (read from its namelist.input), so relocated stations get the location they had during
that event.

Usage examples:
    python make_tslists.py ../runs --stationinfo stationinfo.xls                        (uses default bounds)
    python make_tslists.py ../runs --stationinfo stationinfo.xls --bounds 32.5 34.5 -103.0 -101.0

Reading .xls files requires pandas and xlrd (.xlsx: openpyxl).
"""

import argparse
import pathlib
import re
import sys
from collections import namedtuple
from datetime import datetime

import pandas as pd

from make_namelists import get_value

TOWER = ("RTC 200m Tower", "tower", 33.610, -102.050)
BOUNDS = [(32.10, 35.10), (-103.67, -100.11)] # latitudes and longitudes specifying bounding box for mesonet stations in ROI

HEADER = [
    "#-----------------------------------------------#",
    "# 24 characters for name | pfx |  LAT  |   LON  |",
    "#-----------------------------------------------#",
]
NAME_LEN = 25
PFX_LEN = 5
WRF_DEFAULT_MAX_TS_LOCS = 5

# Previous locations of a relocated station: valid until the date its replacement came online
PastLocation = namedtuple("PastLocation", ["until", "lat", "lon"])


class ItemError(Exception):
    pass


class Station:
    def __init__(self, name, pfx, lat, lon, history):
        self.name = name
        self.pfx = pfx
        self.lat = lat
        self.lon = lon
        self.history = history

    def location_at(self, time):
        """(name, pfx, lat, lon) at the given time, using the earliest relocation that happened after it."""
        later = [past for past in self.history if past.until > time]
        if later:
            past = min(later, key=lambda p: p.until)
            return (self.name, self.pfx, past.lat, past.lon)
        return (self.name, self.pfx, self.lat, self.lon)


def _normalize(column):
    return re.sub(r"[^a-z0-9]", "", str(column).lower())


def find_column(table, pattern, required=True):
    """Column whose lowercased alphanumeric name fully matches pattern (tolerates spacing/punctuation changes)."""
    matches = [column for column in table.columns if re.fullmatch(pattern, _normalize(column))]
    if len(matches) > 1:
        raise ItemError(f"several stationinfo columns match {pattern!r}: {matches}")
    if not matches:
        if required:
            raise ItemError(f"no stationinfo column matches {pattern!r} (columns: {list(table.columns)})")
        return None
    return matches[0]


def _numbers(cell):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", cell)]


def read_stationinfo(path):
    """Stations from a West Texas Mesonet stationinfo spreadsheet (one row per station)."""
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path, dtype=str)
    else:
        table = pd.read_excel(path, dtype=str)
    table = table.fillna("")

    name_col = find_column(table, r"location")
    id_col = find_column(table, r"id")
    lat_col = find_column(table, r"lat(itude)?decimals?")
    lon_col = find_column(table, r"lon(g|gitude)?decimals?")
    prev_id_col = find_column(table, r"pre(v|vious|voius)ids?", required=False)
    prev_lat_col = find_column(table, r"prev(ious)?lat(itude)?decimals?", required=False)
    prev_lon_col = find_column(table, r"prev(ious)?lon(g|gitude)?decimals?", required=False)
    date_col = find_column(table, r"datesofrelocation\w*", required=False)

    stations = []
    for row_number, row in table.iterrows():
        station_id = row[id_col].strip()
        if not station_id:
            continue
        where = f"{path.name} row {row_number + 2} ({station_id})"
        # Data files are named by the 4 letter ID; a trailing number only marks a relocation (REES2 -> REES)
        base_id = re.fullmatch(r"([A-Za-z]{4})\d*", station_id)
        pfx = (base_id.group(1) if base_id else station_id).upper()
        name = row[name_col].strip() or pfx
        if len(name) > NAME_LEN:
            print(f"note: {where}: name '{name}' truncated to {NAME_LEN} characters")
            name = name[:NAME_LEN].rstrip()
        try:
            lat, lon = float(row[lat_col]), float(row[lon_col])
        except ValueError:
            raise ItemError(f"{where}: unreadable decimal latitude/longitude {row[lat_col]!r}, {row[lon_col]!r}")

        history = []
        if prev_id_col and row[prev_id_col].strip():
            prev_ids = re.findall(r"[A-Za-z]+\d*", row[prev_id_col])
            prev_lats = _numbers(row[prev_lat_col]) if prev_lat_col else []
            prev_lons = _numbers(row[prev_lon_col]) if prev_lon_col else []
            dates = re.findall(r"\d{8}", row[date_col]) if date_col else []
            if not len(prev_ids) == len(prev_lats) == len(prev_lons) == len(dates):
                raise ItemError(f"{where}: previous IDs {prev_ids}, latitudes {prev_lats}, longitudes {prev_lons}, "
                                f"and relocation dates {dates} don't line up")
            for date, prev_lat, prev_lon in zip(dates, prev_lats, prev_lons):
                history.append(PastLocation(datetime.strptime(date, "%Y%m%d"), prev_lat, prev_lon))
        stations.append(Station(name, pfx, lat, lon, history))
    return stations


def in_bounds(lat, lon, bounds):
    (lat_min, lat_max), (lon_min, lon_max) = bounds
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def make_tslist(locations):
    lines = list(HEADER)
    seen = set()
    for longname, pfx, lat, lon in locations:
        if len(longname) > NAME_LEN:
            raise ItemError(f"Location name '{longname}' longer than {NAME_LEN} characters")
        if not re.fullmatch(rf"\S{{1,{PFX_LEN}}}", pfx):
            raise ItemError(f"Prefix '{pfx}' for location '{longname}' must be 1-{PFX_LEN} characters without spaces")
        if pfx in seen:
            raise ItemError(f"Prefix '{pfx}' is used by more than one location")
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ItemError(f"Latitude/longitude {lat}, {lon} for location '{longname}' out of range")
        seen.add(pfx)
        lines.append(f"{longname:<{NAME_LEN}} {pfx:<{PFX_LEN}} {lat:7.3f} {lon:8.3f}")
    return lines


def find_run_dirs(outdir):
    if not outdir.is_dir():
        sys.exit(f"error: {outdir} is not a directory")
    subdirs = sorted(d for d in outdir.iterdir() if d.is_dir())
    run_dirs = [d for d in subdirs if (d / "namelist.input").is_file()]
    for d in subdirs:
        if d not in run_dirs:
            print(f"note: skipping {d} (no namelist.input)")
    if not run_dirs:
        sys.exit(f"error: no run directories (subdirectories with a namelist.input) in {outdir}; run make_namelists.py first")
    return run_dirs


def _first_int(lines, group, key, default=None):
    value = get_value(lines, group, key)
    return int(value.split(",")[0]) if value else default


def run_start(lines):
    keys = ("start_year", "start_month", "start_day", "start_hour")
    year, month, day, hour = (_first_int(lines, "time_control", key) for key in keys)
    if None in (year, month, day, hour):
        raise ItemError("namelist.input is missing start_year/month/day/hour in &time_control")
    return datetime(year, month, day, hour)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outdir", type=pathlib.Path, help="runs directory whose run subdirectories get a tslist")
    parser.add_argument("--stationinfo", type=pathlib.Path, required=True,
                        help="mesonet station info spreadsheet (stationinfo.xls)")
    (lat_min, lat_max), (lon_min, lon_max) = BOUNDS
    parser.add_argument("--bounds", type=float, nargs=4, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                        default=[lat_min, lat_max, lon_min, lon_max],
                        help=f"mesonet station bounding box (default: {lat_min} {lat_max} {lon_min} {lon_max})")
    parser.add_argument("--force", action="store_true", help="overwrite existing tslist files")
    args = parser.parse_args(argv)
    lat_min, lat_max, lon_min, lon_max = args.bounds
    if not (lat_min < lat_max and lon_min < lon_max):
        parser.error("--bounds needs LAT_MIN < LAT_MAX and LON_MIN < LON_MAX")
    args.bounds = [(lat_min, lat_max), (lon_min, lon_max)]
    return args


def main(argv=None):
    args = parse_args(argv)
    if not in_bounds(TOWER[2], TOWER[3], args.bounds):
        print(f"note: tower ({TOWER[2]}, {TOWER[3]}) is outside the bounding box; it is included anyway")
    try:
        stations = read_stationinfo(args.stationinfo)
    except ItemError as e:
        sys.exit(f"error: {e}")

    jobs = []
    included = {}
    for run_dir in find_run_dirs(args.outdir):
        path = run_dir / "tslist"
        if path.exists() and not args.force:
            sys.exit(f"error: {path} exists (use --force to overwrite)")
        namelist = (run_dir / "namelist.input").read_text().splitlines()
        try:
            start = run_start(namelist)
            locations = [TOWER] + [loc for loc in (s.location_at(start) for s in stations)
                                   if in_bounds(loc[2], loc[3], args.bounds)]
            lines = make_tslist(locations)
        except ItemError as e:
            sys.exit(f"error: {run_dir}: {e}")
        # WRF silently ignores locations beyond max_ts_locs
        max_ts_locs = _first_int(namelist, "domains", "max_ts_locs", WRF_DEFAULT_MAX_TS_LOCS)
        if len(locations) > max_ts_locs:
            sys.exit(f"error: {run_dir}: {len(locations)} locations but max_ts_locs = {max_ts_locs} in namelist.input")
        jobs.append((path, lines, len(locations), max_ts_locs))
        included[start] = [loc[1] for loc in locations[1:]]

    # Only write once every tslist has been generated without errors
    for path, lines, count, max_ts_locs in jobs:
        with open(path, "w", newline="\n") as f:  # LF endings even when run on Windows
            f.write("\n".join(lines) + "\n")
        print(f"{path}: tower + {count - 1} stations (max_ts_locs = {max_ts_locs})")
    for start, pfxs in sorted(included.items()):
        print(f"stations for runs starting {start:%Y-%m-%d_%H}: {' '.join(pfxs) or '(none)'}")


if __name__ == "__main__":
    main()
