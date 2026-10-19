"""
Generate one namelist.input per (event, PBL scheme) run from namelist.input.template.

Only the run dates, bl_pbl_physics, and sf_sfclay_physics (plus any scheme-specific extras)
are changed; everything else is copied from the template so all runs share the same setup.
Each run gets its own directory: OUTDIR/<event>_<scheme>/namelist.input

Usage examples:
    python make_namelists.py ../runs                                  (all events, core schemes)
    python make_namelists.py ../runs --events 20131210 20140125 --schemes ysu mynn acm2 myj
    python make_namelists.py ../runs_test --events 20131210 --schemes ysu --run-hours 3
"""

import argparse
import pathlib
import re
import sys
from datetime import datetime, timedelta

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

DEFAULT_TEMPLATE = pathlib.Path(__file__).parent / "namelist.input.template"

_GROUP_START = re.compile(r"^\s*&(\w+)\s*$")
_GROUP_END = re.compile(r"^\s*/\s*$")


class TemplateError(Exception):
    pass


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
    max_dom_str = get_value(lines, "domains", "max_dom")
    if max_dom_str is None:
        raise TemplateError("template &domains has no max_dom")
    max_dom = int(max_dom_str)
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outdir", type=pathlib.Path, help="directory to create <event>_<scheme>/namelist.input in")
    parser.add_argument("--template", type=pathlib.Path, default=DEFAULT_TEMPLATE,
                        help="full namelist.input template (default: %(default)s)")
    parser.add_argument("--events", nargs="+", default=EVENTS, metavar="YYYYMMDD",
                        help=f"event (morning) dates (default: all {len(EVENTS)})")
    parser.add_argument("--schemes", nargs="+", default=CORE_SCHEMES, choices=sorted(SCHEMES),
                        help=f"PBL schemes (default: {' '.join(CORE_SCHEMES)})")
    parser.add_argument("--run-hours", type=int, default=None,
                        help="override the template run length, e.g. 3 for a test run")
    parser.add_argument("--force", action="store_true", help="overwrite existing namelist.input files")
    args = parser.parse_args(argv)
    for event in args.events:
        try:
            datetime.strptime(event, "%Y%m%d")
        except ValueError:
            parser.error(f"event {event!r} is not a YYYYMMDD date")
    return args


def main(argv=None):
    args = parse_args(argv)
    template_lines = args.template.read_text().splitlines()

    jobs = []
    for event in args.events:
        for scheme in args.schemes:
            path = args.outdir / f"{event}_{scheme}" / "namelist.input"
            if path.exists() and not args.force:
                sys.exit(f"error: {path} exists (use --force to overwrite)")
            try:
                lines, start, end = make_namelist(template_lines, event, scheme, args.run_hours)
            except TemplateError as e:
                sys.exit(f"error: {args.template}: {e}")
            jobs.append((path, lines, event, scheme, start, end))

    # Only write once every namelist has been generated without errors
    for path, lines, event, scheme, start, end in jobs:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="\n") as f:  # LF endings even when run on Windows
            f.write("\n".join(lines) + "\n")
        config = SCHEMES[scheme]
        print(f"{event}_{scheme:<8s} {start:%Y-%m-%d_%H} -> {end:%Y-%m-%d_%H}  "
              f"bl_pbl_physics={config['pbl']} sf_sfclay_physics={config['sfclay']}  {path}")

    iofields = get_value(template_lines, "time_control", "iofields_filename")
    if iofields:
        print(f"note: template sets iofields_filename = {iofields}; that file must be in each run directory")


if __name__ == "__main__":
    main()
