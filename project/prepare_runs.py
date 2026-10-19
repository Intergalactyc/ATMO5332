"""
Create and prepare one WRF run directory per (event, PBL scheme): OUTDIR/<event>_<scheme>/

Each run directory gets:
    namelist.input  from configs/namelist.input.template; only the run dates, bl_pbl_physics, and sf_sfclay_physics
                    (plus any scheme-specific extras) change, so all runs share the same setup
    tslist          the 200 m tower plus the mesonet stations inside a lat/lon bounding box, with station coordinates
                    as of the run's start time (relocated stations get the location they had during that event)
    iofields.txt    copied from configs/ when the template sets iofields_filename
and, optionally (on HPCC):
    --wrf-run DIR   symlinks to the contents of the WRF run directory (real.exe, wrf.exe, lookup tables, ...)
    --met-em DIR    symlinks to the met_em files real.exe needs (d01 at every boundary time, nests at the start);
                    "{event}" in DIR is replaced by the event date, e.g. /lustre/work/USER/pbl/wps/{event}

Station lists can be either:
    - the official station location list (meta/wtm_2013_station_locations.csv; columns archive_name, Location,
      latitude, longitude, ...): archive_name is used as the tslist name, and a unique prefix of up to 5 characters
      is made from it
    - a WTM stationinfo.xls (columns Location, Lat-decimal, Long.-decimal, ID, plus relocation history): the
      4 letter ID is used as the prefix and Location as the name

Usage examples:
    python prepare_runs.py ../runs                                  (all events, core schemes)
    python prepare_runs.py ../runs --events 20131210 20140125 --schemes ysu mynn acm2 myj
    python prepare_runs.py ../runs_test --events 20131210 --schemes ysu --run-hours 3
    python prepare_runs.py ../runs --wrf-run /lustre/work/USER/WRFV4/run --met-em /lustre/work/USER/pbl/wps/{event}
    python prepare_runs.py ../runs --bounds 32.5 34.5 -103.0 -101.0 --stationinfo stationinfo.xls

Nothing is written unless every run directory can be prepared without errors. Existing files are only replaced
with --force; files that aren't created here (e.g. WRF output from an earlier run) are never touched.

Requires pandas (and xlrd for .xls, openpyxl for .xlsx).
"""

import argparse
import fnmatch
import os
import pathlib
import re
import shutil
import sys
import tempfile
from collections import Counter, namedtuple
from datetime import datetime, timedelta

import pandas as pd

ROOT = pathlib.Path(__file__).parent
DEFAULT_TEMPLATE = ROOT / "configs" / "namelist.input.template"
DEFAULT_STATIONINFO = ROOT / "meta" / "wtm_2013_station_locations.csv"

# Events are named by the morning (UTC date) of the LLJ; each run starts 18 UTC the day prior
EVENTS = [
    "20131210",  # key
    "20140125",  # key
    "20131216",  # backup
    "20140122",  # backup
    "20140214",  # backup
]
INIT_OFFSET = timedelta(hours=-6)  # 00 UTC on the event date minus 6 h = 18 UTC the day prior

# bl_pbl_physics / sf_sfclay_physics pairs
# sfclay 1 is used wherever the scheme accepts it.
# "extra" lists any other entries a scheme needs as (group, key, value, per_domain)
SCHEMES = {
    # Core comparison
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
    "sms3dtke": {"pbl": 0,  "sfclay": 1, "extra": [("dynamics", "km_opt", 5, True), ("dynamics", "diff_opt", 2, True)]}, # SMS-3DTKE: 3D scale-adaptive TKE closure
    "myj":      {"pbl": 2,  "sfclay": 2},
    "gfs":      {"pbl": 3,  "sfclay": 3},
    "qnse":     {"pbl": 4,  "sfclay": 4},
    "temf":     {"pbl": 10, "sfclay": 10},
}
CORE_SCHEMES = ["ysu", "mynn", "acm2"]

TOWER = ("RTC 200m Tower", "tower", 33.610, -102.050)
BOUNDS = [(32.10, 35.10), (-103.67, -100.11)] # latitudes and longitudes specifying bounding box for mesonet stations in ROI

# WRF (share/wrf_tsin.F, v4.5.1) reads the second header line with FMT='(32X,A7,2X,A7)' and needs "LAT" and "LON"
# in those fields, then reads each location with FMT='(A25,1X,A5,1X,F7.3,1X,F8.3)'
TSLIST_HEADER = [
    "#-----------------------------------------------#",
    "# 24 characters for name | pfx |  LAT  |   LON  |",
    "#-----------------------------------------------#",
]
NAME_LEN = 25
PFX_LEN = 5
WRF_DEFAULT_MAX_TS_LOCS = 5

# Contents of the WRF run directory that are not linked: per-run inputs this script writes or links itself,
# and output that may be left over from earlier runs there
WRF_RUN_SKIP = ["namelist.*", "tslist", "iofields*", "met_em.*", "wrfinput_*", "wrfbdy_*", "wrflowinp_*",
                "wrffdda_*", "wrfout_*", "wrfrst_*", "rsl.*", "*.TS", "*.log"]

# Previous locations of a relocated station: valid until the date its replacement came online
PastLocation = namedtuple("PastLocation", ["until", "lat", "lon"])

_GROUP_START = re.compile(r"^\s*&(\w+)\s*$")
_GROUP_END = re.compile(r"^\s*/\s*$")


class TemplateError(Exception):
    pass


class ItemError(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------------
# namelist.input

def find_group(lines, group):
    """Return (start, end) line indices of &group ... / in the namelist lines."""
    start = None
    for i, line in enumerate(lines):
        m = _GROUP_START.match(line)
        if start is None:
            if m and m.group(1).lower() == group:
                start = i
        elif _GROUP_END.match(line):
            return start, i
    raise TemplateError(f"template has no complete &{group} group")


def _entry_lines(lines, group, key):
    start, end = find_group(lines, group)
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)(.*?)\s*$", re.IGNORECASE)
    hits = [i for i in range(start + 1, end) if pattern.match(lines[i])]
    if len(hits) > 1:
        raise TemplateError(f"{key} appears {len(hits)} times in &{group}")
    return pattern, hits, start, end


def get_value(lines, group, key):
    pattern, hits, _, _ = _entry_lines(lines, group, key)
    if not hits:
        return None
    return pattern.match(lines[hits[0]]).group(2).rstrip(",").strip()


def set_value(lines, group, key, value):
    """Replace the value of key in &group (keeping its alignment), or add the entry if missing."""
    pattern, hits, start, end = _entry_lines(lines, group, key)
    if hits:
        prefix, old = pattern.match(lines[hits[0]]).groups()
        trailing = "," if old.endswith(",") else ""
        lines[hits[0]] = prefix + value + trailing
        return
    # New entry: align the "=" with the first existing entry in the group
    indent, width, trailing = " ", len(key) + 1, ""
    for line in lines[start + 1:end]:
        m = re.match(r"^(\s*)(\w+\s*)=(.*)$", line)
        if m:
            indent, width = m.group(1), len(m.group(2))
            trailing = "," if m.group(3).strip().endswith(",") else ""
            break
    lines.insert(end, f"{indent}{key.ljust(max(width, len(key) + 1))}= {value}{trailing}")


def _first_int(lines, group, key, default=None):
    value = get_value(lines, group, key)
    return int(value.split(",")[0]) if value else default


def per_domain(value, max_dom):
    return ", ".join([str(value)] * max_dom)


def run_length(lines, run_hours):
    """Forecast length from --run-hours if given, otherwise from run_days/hours/minutes/seconds in the template."""
    if run_hours is not None:
        return timedelta(hours=run_hours)
    parts = {}
    for unit in ("days", "hours", "minutes", "seconds"):
        value = get_value(lines, "time_control", "run_" + unit)
        parts[unit] = int(value) if value else 0
    length = timedelta(**parts)
    if length <= timedelta(0):
        raise TemplateError("template has no run length; set run_hours in &time_control or pass --run-hours")
    return length


def make_namelist(template_lines, event, scheme, run_hours=None):
    lines = list(template_lines)
    max_dom = _first_int(lines, "domains", "max_dom")
    if max_dom is None:
        raise TemplateError("template &domains has no max_dom")
    find_group(lines, "physics")  # fail early if the template has no &physics group

    start = datetime.strptime(event, "%Y%m%d") + INIT_OFFSET
    end = start + run_length(lines, run_hours)

    if run_hours is not None:
        set_value(lines, "time_control", "run_days", "0")
        set_value(lines, "time_control", "run_hours", str(run_hours))
        set_value(lines, "time_control", "run_minutes", "0")
        set_value(lines, "time_control", "run_seconds", "0")
    for prefix, date in (("start", start), ("end", end)):
        set_value(lines, "time_control", f"{prefix}_year", per_domain(date.year, max_dom))
        set_value(lines, "time_control", f"{prefix}_month", per_domain(f"{date.month:02d}", max_dom))
        set_value(lines, "time_control", f"{prefix}_day", per_domain(f"{date.day:02d}", max_dom))
        set_value(lines, "time_control", f"{prefix}_hour", per_domain(f"{date.hour:02d}", max_dom))

    config = SCHEMES[scheme]
    set_value(lines, "physics", "bl_pbl_physics", per_domain(config["pbl"], max_dom))
    set_value(lines, "physics", "sf_sfclay_physics", per_domain(config["sfclay"], max_dom))
    for group, key, value, is_per_domain in config.get("extra", []):
        set_value(lines, group, key, per_domain(value, max_dom) if is_per_domain else str(value))

    return lines, start, end


# ---------------------------------------------------------------------------------------------------------------------
# tslist

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


def _station_name(name, where):
    if len(name) > NAME_LEN:
        print(f"note: {where}: name '{name}' truncated to {NAME_LEN} characters")
        name = name[:NAME_LEN].rstrip()
    return name


def _lat_lon(row, lat_col, lon_col, where):
    try:
        return float(row[lat_col]), float(row[lon_col])
    except ValueError:
        raise ItemError(f"{where}: unreadable decimal latitude/longitude {row[lat_col]!r}, {row[lon_col]!r}")


def short_prefixes(names):
    """Unique prefixes of up to PFX_LEN characters: the first 5 letters, or the first 4 plus the last on a clash."""
    letters = [re.sub(r"[^A-Za-z0-9]", "", name).upper() for name in names]
    first = Counter(word[:PFX_LEN] for word in letters)
    pfxs = [word[:PFX_LEN] if first[word[:PFX_LEN]] == 1 else word[:PFX_LEN - 1] + word[-1] for word in letters]
    clashes = sorted(pfx for pfx, count in Counter(pfxs).items() if count > 1)
    if clashes:
        raise ItemError(f"could not make unique {PFX_LEN} character prefixes: {clashes}")
    return pfxs


def read_archive_list(table, path):
    """Stations from the official location list: one row per station, archive_name as the identifier."""
    archive_col = find_column(table, r"archivename")
    lat_col = find_column(table, r"lat(itude)?")
    lon_col = find_column(table, r"lon(g|gitude)?")
    rows = [(row_number, row) for row_number, row in table.iterrows() if row[archive_col].strip()]
    pfxs = short_prefixes([row[archive_col].strip() for _, row in rows])
    stations = []
    for (row_number, row), pfx in zip(rows, pfxs):
        archive_name = row[archive_col].strip()
        where = f"{path.name} row {row_number + 2} ({archive_name})"
        lat, lon = _lat_lon(row, lat_col, lon_col, where)
        stations.append(Station(_station_name(archive_name, where), pfx, lat, lon, []))
    return stations


def read_wtm_stationinfo(table, path):
    """Stations from a West Texas Mesonet stationinfo spreadsheet (one row per station, with relocation history)."""
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
        name = _station_name(row[name_col].strip() or pfx, where)
        lat, lon = _lat_lon(row, lat_col, lon_col, where)

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


def read_stationinfo(path):
    """Stations from either supported station list format (.csv, .xls, or .xlsx), detected from its columns."""
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path, dtype=str)
    else:
        table = pd.read_excel(path, dtype=str)
    table = table.fillna("")
    if find_column(table, r"archivename", required=False):
        return read_archive_list(table, path)
    if find_column(table, r"id", required=False):
        return read_wtm_stationinfo(table, path)
    raise ItemError(f"{path.name}: unrecognized station list (needs an archive_name or ID column; "
                    f"columns: {list(table.columns)})")


def in_bounds(lat, lon, bounds):
    (lat_min, lat_max), (lon_min, lon_max) = bounds
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def make_tslist(locations):
    lines = list(TSLIST_HEADER)
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


# ---------------------------------------------------------------------------------------------------------------------
# Other run directory inputs

def iofields_source(template_path, template_lines):
    """The file named by iofields_filename in the template (looked up next to the template), or None."""
    value = get_value(template_lines, "time_control", "iofields_filename")
    if not value:
        return None
    names = {name.strip().strip("'\"") for name in value.split(",") if name.strip()}
    if len(names) != 1:
        raise TemplateError(f"iofields_filename should name one file for all domains, got {sorted(names)}")
    source = template_path.parent / names.pop()
    if not source.is_file():
        raise TemplateError(f"template sets iofields_filename but {source} does not exist")
    return source


def wrf_run_links(wrf_run):
    """Names in the WRF run directory to symlink into each run directory."""
    if not wrf_run.is_dir():
        raise TemplateError(f"--wrf-run {wrf_run} is not a directory")
    names = sorted(entry.name for entry in wrf_run.iterdir()
                   if not entry.is_dir() and not any(fnmatch.fnmatch(entry.name, p) for p in WRF_RUN_SKIP))
    for exe in ("real.exe", "wrf.exe"):
        if exe not in names:
            raise TemplateError(f"--wrf-run {wrf_run} has no {exe}")
    return names


def met_em_files(met_em_dir, lines, start, end):
    """met_em files real.exe needs: d01 at every interval_seconds from start to end, nests at the start only."""
    max_dom = _first_int(lines, "domains", "max_dom")
    interval = timedelta(seconds=_first_int(lines, "time_control", "interval_seconds", 10800))
    times = []
    time = start
    while time <= end:
        times.append(time)
        time += interval
    needed = [f"met_em.d01.{t:%Y-%m-%d_%H:%M:%S}.nc" for t in times]
    needed += [f"met_em.d{d:02d}.{start:%Y-%m-%d_%H:%M:%S}.nc" for d in range(2, max_dom + 1)]
    missing = [name for name in needed if not (met_em_dir / name).is_file()]
    if missing:
        raise TemplateError(f"{len(missing)} of {len(needed)} met_em files missing from {met_em_dir}, "
                            f"e.g. {missing[0]}")
    return needed


def check_symlinks():
    """Exit before writing anything if this system can't create symlinks (Windows without Developer Mode)."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.symlink(tmp, os.path.join(tmp, "link"))
        except OSError as e:
            sys.exit(f"error: can't create symlinks here ({e}); run with --wrf-run/--met-em on HPCC")


def _replace(path, force):
    """Make way for a new file or link at path; only with --force if something is already there."""
    if path.is_symlink() or path.exists():
        if not force:
            sys.exit(f"error: {path} exists (use --force to overwrite)")
        if path.is_dir() and not path.is_symlink():
            sys.exit(f"error: {path} is a directory; not replacing it")
        path.unlink()


# ---------------------------------------------------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outdir", type=pathlib.Path, help="directory to create the <event>_<scheme> run directories in")
    parser.add_argument("--template", type=pathlib.Path, default=DEFAULT_TEMPLATE,
                        help="full namelist.input template (default: %(default)s)")
    parser.add_argument("--events", nargs="+", default=EVENTS, metavar="YYYYMMDD",
                        help=f"event (morning) dates (default: all {len(EVENTS)})")
    parser.add_argument("--schemes", nargs="+", default=CORE_SCHEMES, choices=sorted(SCHEMES),
                        help=f"PBL schemes (default: {' '.join(CORE_SCHEMES)})")
    parser.add_argument("--run-hours", type=int, default=None,
                        help="override the template run length, e.g. 3 for a test run")
    parser.add_argument("--stationinfo", type=pathlib.Path, default=DEFAULT_STATIONINFO,
                        help="station list: official location .csv or WTM stationinfo.xls (default: %(default)s)")
    (lat_min, lat_max), (lon_min, lon_max) = BOUNDS
    parser.add_argument("--bounds", type=float, nargs=4, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
                        default=[lat_min, lat_max, lon_min, lon_max],
                        help=f"mesonet station bounding box (default: {lat_min} {lat_max} {lon_min} {lon_max})")
    parser.add_argument("--wrf-run", type=pathlib.Path, default=None,
                        help="WRF run directory to symlink executables and lookup tables from")
    parser.add_argument("--met-em", default=None, metavar="DIR",
                        help="directory with each event's met_em files to symlink; {event} is replaced by the event")
    parser.add_argument("--force", action="store_true", help="replace existing files and links in the run directories")
    args = parser.parse_args(argv)
    for event in args.events:
        try:
            datetime.strptime(event, "%Y%m%d")
        except ValueError:
            parser.error(f"event {event!r} is not a YYYYMMDD date")
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
        template_lines = args.template.read_text().splitlines()
        iofields = iofields_source(args.template, template_lines)
        wrf_links = wrf_run_links(args.wrf_run) if args.wrf_run else []
        stations = read_stationinfo(args.stationinfo)
    except (TemplateError, ItemError) as e:
        sys.exit(f"error: {e}")
    if args.wrf_run or args.met_em:
        check_symlinks()

    # Build everything in memory first, so nothing is written unless every run can be prepared
    runs = []
    for event in args.events:
        for scheme in args.schemes:
            run_dir = args.outdir / f"{event}_{scheme}"
            try:
                namelist, start, end = make_namelist(template_lines, event, scheme, args.run_hours)
                locations = [TOWER] + [loc for loc in (s.location_at(start) for s in stations)
                                       if in_bounds(loc[2], loc[3], args.bounds)]
                tslist = make_tslist(locations)
                # WRF silently ignores locations beyond max_ts_locs
                max_ts_locs = _first_int(namelist, "domains", "max_ts_locs", WRF_DEFAULT_MAX_TS_LOCS)
                if len(locations) > max_ts_locs:
                    raise ItemError(f"{len(locations)} tslist locations but max_ts_locs = {max_ts_locs} in the template")
                met_em_dir = pathlib.Path(args.met_em.replace("{event}", event)) if args.met_em else None
                met_em = met_em_files(met_em_dir, namelist, start, end) if met_em_dir else []
            except (TemplateError, ItemError) as e:
                sys.exit(f"error: {run_dir}: {e}")
            runs.append({"dir": run_dir, "event": event, "scheme": scheme, "start": start, "end": end,
                         "namelist": namelist, "tslist": tslist, "stations": [loc[1] for loc in locations[1:]],
                         "met_em_dir": met_em_dir, "met_em": met_em})

    # Refuse to overwrite anything without --force before creating any files
    for run in runs:
        names = ["namelist.input", "tslist"] + ([iofields.name] if iofields else []) + wrf_links + run["met_em"]
        for name in names:
            target = run["dir"] / name
            if (target.is_symlink() or target.exists()) and not args.force:
                sys.exit(f"error: {target} exists (use --force to overwrite)")

    for run in runs:
        run_dir = run["dir"]
        run_dir.mkdir(parents=True, exist_ok=True)
        for name, lines in (("namelist.input", run["namelist"]), ("tslist", run["tslist"])):
            _replace(run_dir / name, args.force)
            with open(run_dir / name, "w", newline="\n") as f:  # LF endings even when run on Windows
                f.write("\n".join(lines) + "\n")
        if iofields:
            _replace(run_dir / iofields.name, args.force)
            shutil.copyfile(iofields, run_dir / iofields.name)
        for name in wrf_links:
            _replace(run_dir / name, args.force)
            os.symlink(args.wrf_run.resolve() / name, run_dir / name)
        for name in run["met_em"]:
            _replace(run_dir / name, args.force)
            os.symlink(run["met_em_dir"].resolve() / name, run_dir / name)

        config = SCHEMES[run["scheme"]]
        extras = [f"tslist (tower + {len(run['stations'])} stations)"]
        if iofields:
            extras.append(iofields.name)
        if wrf_links:
            extras.append(f"{len(wrf_links)} WRF run links")
        if run["met_em"]:
            extras.append(f"{len(run['met_em'])} met_em links")
        print(f"{run['event']}_{run['scheme']:<8s} {run['start']:%Y-%m-%d_%H} -> {run['end']:%Y-%m-%d_%H}  "
              f"bl_pbl_physics={config['pbl']} sf_sfclay_physics={config['sfclay']}  {run_dir}: "
              f"namelist.input, {', '.join(extras)}")

    for start in sorted({run["start"] for run in runs}):
        pfxs = next(run["stations"] for run in runs if run["start"] == start)
        print(f"stations for runs starting {start:%Y-%m-%d_%H} ({len(pfxs)}): {' '.join(pfxs) or '(none)'}")
    if not args.wrf_run or not args.met_em:
        print("note: run directories still need the WRF executables/tables (--wrf-run) and met_em files (--met-em) "
              "before real.exe can run")


if __name__ == "__main__":
    main()
