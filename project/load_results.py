"""
load_results.py
Module for loading results for analysis

Usage:
    import load_results
    runs = load_results.load_all()                # {(date, scheme): RunOutput}; instant, data is read lazily
    run = runs["20131210", "ysu"]                 # dates may be "YYYYMMDD", "YYYY-MM-DD", date or datetime
    run = load_results.load_run("20131210", "ysu")  # same thing, one run
    abern = run.stations["ABERN"]                 # station map: prefix -> Station
    abern.df[["t", "ws", "wd"]]                   # surface time series (.TS), indexed by valid time
    abern.ws, abern.z                             # profiles: time x model level (1 = lowest)
    run.tower.column.u                            # tower_column.nc (wrfout column at the tower grid point)

Winds: WRF (4.5.2) already rotates the time series winds (.TS u/v and the .UU/.VV profiles) to earth-relative
components with COSALPHA/SINALPHA (share/wrf_timeseries.F), and pack_results.py applies the same rotation to
tower_column.nc. So u is west-east and v is south-north everywhere here; no further rotation is needed.
Wind direction (wd) is meteorological: degrees the wind blows FROM, clockwise from north.

Each Station's data is read from disk the first time it is accessed and then cached, so loading every run is
cheap and memory only grows with what is actually used (all of it would be ~22 GB).
"""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
import os
import re
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
from netCDF4 import Dataset

RESULTS_DIR = Path(__file__).parent / "results"
if not os.path.exists(RESULTS_DIR):
    raise Exception("Results directory does not exist!")

DOMAIN = "d03"
TOWER_PREFIX = "tower"  # prefix of the tower in tslist (prepare_runs.TOWER)

# Columns of the .TS file (WRF README.tslist); id/id_tsloc/ix/iy are dropped after reading
TS_COLUMNS = ["id", "hour", "id_tsloc", "ix", "iy",
              "t",       # 2 m temperature (K)
              "q",       # 2 m vapor mixing ratio (kg/kg)
              "u",       # 10 m west-east wind (m/s, earth-relative)
              "v",       # 10 m south-north wind (m/s, earth-relative)
              "psfc",    # surface pressure (Pa)
              "glw",     # downward longwave at surface (W/m2)
              "gsw",     # net shortwave at surface (W/m2)
              "hfx",     # surface sensible heat flux (W/m2)
              "lh",      # surface latent heat flux (W/m2)
              "tsk",     # skin temperature (K)
              "tslb",    # top soil layer temperature (K)
              "rainc",   # accumulated cumulus precipitation (mm)
              "rainnc",  # accumulated grid-scale precipitation (mm)
              "clw"]     # column-integrated cloud water (kg/m2)

# "Abernathy   3  2 ABERN ( 33.875,-101.757) ( 195, 213) ( 33.876,-101.762) 1011.2 meters  2013-12-09_18:00:00"
_NUM = r"\s*([-+\d.]+)\s*"
HEADER_RE = re.compile(rf"^(?P<long_name>.*?)\s+\d+\s+\d+\s+(?P<prefix>\S+)\s+"
                       rf"\({_NUM},{_NUM}\)\s+\(\s*(\d+)\s*,\s*(\d+)\s*\)\s+\({_NUM},{_NUM}\)\s+"
                       rf"(?P<elev>[-+\d.]+)\s+meters\s+(?P<start>\S+)")


def wind_speed(u, v):
    return np.hypot(u, v)


def wind_direction(u, v):
    """Meteorological direction (degrees the wind blows from, clockwise from north) of earth-relative u, v."""
    return np.degrees(np.arctan2(-u, -v)) % 360


def _as_date(d) -> date:
    """Accept date, datetime, or "YYYYMMDD" / "YYYY-MM-DD" strings."""
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    s = str(d).replace("-", "")
    return datetime.strptime(s, "%Y%m%d").date()


def _read_numeric(path: Path, ncols: int) -> np.ndarray:
    """All rows after the one-line header of a WRF time series file, as an (ntimes, ncols) array."""
    with open(path) as f:
        f.readline()
        return np.array(f.read().split(), dtype=float).reshape(-1, ncols)


class Station:
    """One tslist location of one run. Metadata comes from the .TS header; data are read on first access."""

    def __init__(self, directory: Path, prefix: str):
        self.directory = Path(directory)
        self.prefix = prefix
        header = self._path("TS").open().readline()
        m = HEADER_RE.match(header)
        if m is None:
            raise ValueError(f"Unrecognized time series header in {self._path('TS')}: {header!r}")
        g = m.groups()
        self.name : str = m["prefix"]  # short name, e.g. "ABERN"
        self.long_name : str = m["long_name"].strip()  # e.g. "Abernathy"
        self.target_latitude, self.target_longitude = float(g[2]), float(g[3])  # requested in tslist
        self.grid_i, self.grid_j = int(g[4]), int(g[5])  # 1-based mass grid point WRF used
        self.latitude : float = float(g[6])  # degrees, model grid point
        self.longitude : float = float(g[7])  # degrees, model grid point
        self.elevation : float = float(m["elev"])  # model terrain height at the grid point, meters ASL
        self.start : datetime = datetime.strptime(m["start"], "%Y-%m-%d_%H:%M:%S")  # run start time

    def __repr__(self):
        return (f"{type(self).__name__}({self.name!r}, {self.long_name!r}, lat={self.latitude}, "
                f"lon={self.longitude}, elev={self.elevation} m)")

    def _path(self, ext: str) -> Path:
        return self.directory / f"{self.prefix}.{DOMAIN}.{ext}"

    def _times(self, hours: np.ndarray) -> pd.DatetimeIndex:
        # hours are written with 6 decimals (e.g. 0.001389 for 5 s), so round to the nearest second
        return pd.DatetimeIndex(self.start + pd.to_timedelta(hours * 3600, unit="s").round("1s"), name="time")

    def _profile(self, ext: str) -> pd.DataFrame:
        with open(self._path(ext)) as f:
            f.readline()
            ncols = len(f.readline().split())
        data = _read_numeric(self._path(ext), ncols)
        levels = pd.RangeIndex(1, ncols, name="level")  # model mass levels, 1 = lowest
        return pd.DataFrame(data[:, 1:], index=self._times(data[:, 0]), columns=levels)

    # surface time series
    @cached_property
    def df(self) -> pd.DataFrame:
        """Surface time series from .TS (see TS_COLUMNS), plus ws and wd of the 10 m wind."""
        data = _read_numeric(self._path("TS"), len(TS_COLUMNS))
        df = pd.DataFrame(data, columns=TS_COLUMNS, index=self._times(data[:, 1]))
        df = df.drop(columns=["id", "id_tsloc", "ix", "iy"])  # constant for a station
        df["ws"] = wind_speed(df["u"], df["v"])
        df["wd"] = wind_direction(df["u"], df["v"])
        return df

    # profiles (time x level)
    @cached_property
    def gph(self) -> pd.DataFrame:
        """Geopotential height of the mass levels (m ASL), .PH"""
        return self._profile("PH")

    @cached_property
    def z(self) -> pd.DataFrame:
        """Height of the mass levels above ground (m AGL): gph - elevation."""
        return self.gph - self.elevation

    @cached_property
    def p(self) -> pd.DataFrame:
        """Pressure (Pa), .PR"""
        return self._profile("PR")

    @cached_property
    def qv(self) -> pd.DataFrame:
        """Water vapor mixing ratio (kg/kg), .QV"""
        return self._profile("QV")

    @cached_property
    def theta(self) -> pd.DataFrame:
        """Potential temperature (K), .TH"""
        return self._profile("TH")

    @cached_property
    def u(self) -> pd.DataFrame:
        """West-east wind (m/s, earth-relative), .UU"""
        return self._profile("UU")

    @cached_property
    def v(self) -> pd.DataFrame:
        """South-north wind (m/s, earth-relative), .VV"""
        return self._profile("VV")

    @cached_property
    def w(self) -> pd.DataFrame:
        """Vertical wind (m/s), .WW"""
        return self._profile("WW")

    @cached_property
    def ws(self) -> pd.DataFrame:
        """Horizontal wind speed (m/s)"""
        return wind_speed(self.u, self.v)

    @cached_property
    def wd(self) -> pd.DataFrame:
        """Wind direction (degrees from, meteorological)"""
        return wind_direction(self.u, self.v)

    def unload(self):
        """Drop cached data to free memory (it is re-read on next access)."""
        for name in ("df", "gph", "z", "p", "qv", "theta", "u", "v", "w", "ws", "wd", "column"):
            self.__dict__.pop(name, None)


@dataclass
class TowerColumn:
    """tower_column.nc: the d03 wrfout column at the tower grid point (every history output, 15 min).
    Profiles are time x level DataFrames; level is 1-based (1 = lowest), level_w has one more level than level."""
    z : pd.DataFrame  # mass level heights, m AGL
    z_w : pd.DataFrame  # w level heights, m AGL
    u : pd.DataFrame  # west-east wind, m/s (earth-relative)
    v : pd.DataFrame  # south-north wind, m/s (earth-relative)
    theta : pd.DataFrame  # potential temperature, K
    p : pd.DataFrame  # pressure, Pa
    qv : pd.DataFrame  # water vapor mixing ratio, kg/kg
    exch_h : pd.DataFrame  # exchange coefficient for heat (w levels; 0 at the ground), m2/s
    exch_m : pd.DataFrame  # exchange coefficient for momentum (w levels; 0 at the ground), m2/s
    sfc : pd.DataFrame  # time series: pblh (m), ust (m/s), hfx (W/m2), lh (W/m2)
    tke_pbl : Optional[pd.DataFrame] = None  # TKE, m2/s2; only for TKE schemes. On w levels, except for MYNN, where
                                             # it is QKE / 2 on mass levels (pack_results.py; TKE_PBL is all zero)


ACM2 = 7  # bl_pbl_physics of ACM2


def pbl_physics(directory: Path) -> Optional[int]:
    """bl_pbl_physics of the run (first domain's value) from its namelist.input, or None if not found."""
    path = Path(directory) / "namelist.input"
    if not path.is_file():
        return None
    m = re.search(r"^\s*bl_pbl_physics\s*=\s*(\d+)", path.read_text(), re.MULTILINE | re.IGNORECASE)
    return int(m[1]) if m else None


def load_tower_column(path: Path, pbl: Optional[int] = None) -> TowerColumn:
    """Read tower_column.nc. pbl is the run's bl_pbl_physics: for ACM2 the exchange coefficients are moved up
    one w level, since ACM2 stores the coefficient for w level k + 1 at index k (EXCH_H(K) = EDDYZ(K), computed at
    ZF(K), the top of layer K, in module_bl_acm.F), while YSU and MYNN store it at index k with k = 1 the ground."""
    with Dataset(path) as nc:
        nc.set_auto_mask(False)
        tvar = nc.variables["time"]
        base = datetime.strptime(tvar.units.split("since ")[1], "%Y-%m-%d %H:%M:%S")
        times = pd.DatetimeIndex(base + pd.to_timedelta(tvar[:], unit="min").round("1s"), name="time")
        profiles, sfc = {}, {}
        for name, var in nc.variables.items():
            if name == "time":
                continue
            data = np.asarray(var[:], dtype=float)
            if data.ndim == 1:
                sfc[name] = data
            else:
                levels = pd.RangeIndex(1, data.shape[1] + 1, name=var.dimensions[1])
                profiles[name] = pd.DataFrame(data, index=times, columns=levels)
    if pbl == ACM2:
        for name in ("exch_h", "exch_m"):
            shifted = profiles[name].shift(1, axis=1)  # value at index k -> w level k + 1; the top one drops off
            shifted[1] = 0.0  # no exchange coefficient at the ground (surface fluxes come from the surface layer)
            profiles[name] = shifted
    return TowerColumn(**profiles, sfc=pd.DataFrame(sfc, index=times))


class Tower(Station):
    """The tower's tslist point (same data as a Station) plus the tower_column.nc column."""

    @cached_property
    def column(self) -> TowerColumn:
        return load_tower_column(self.directory / "tower_column.nc", pbl_physics(self.directory))


@dataclass
class RunOutput:
    stations : dict[str, Station] = field(repr=False)  # station map: prefix (e.g. "ABERN") -> Station
    tower : Tower = field(repr=False)
    date_obj : date
    scheme : str
    directory : Path = field(repr=False)

    def __repr__(self):
        return f"RunOutput({self.date_obj:%Y%m%d}, {self.scheme!r}, {len(self.stations)} stations)"


class RunMap(dict):
    """Run map returned by load_all: {(date, scheme): RunOutput}. Lookups accept any date form _as_date takes,
    e.g. runs["20131210", "ysu"] or runs[date(2013, 12, 10), "ysu"]."""

    @staticmethod
    def _key(key):
        d, scheme = key
        return _as_date(d), scheme.lower()

    def __getitem__(self, key):
        return super().__getitem__(self._key(key))

    def __contains__(self, key):
        try:
            return super().__contains__(self._key(key))
        except (TypeError, ValueError):
            return False

    def get(self, key, default=None):
        return self[key] if key in self else default

    @property
    def dates(self) -> list[date]:
        return sorted({d for d, _ in self})

    @property
    def schemes(self) -> list[str]:
        return sorted({s for _, s in self})


def identify_runs():
    runs = []
    for spath in sorted(os.listdir(RESULTS_DIR)):
        joined = RESULTS_DIR / spath
        if not os.path.isdir(joined):
            continue
        split = spath.split("_")
        if len(split) != 2:
            continue
        datestr, scheme = split
        try:
            date_obj = datetime.strptime(datestr, "%Y%m%d").date()
        except ValueError:
            continue
        runs.append((joined.resolve(), date_obj, scheme))
    return runs


def load_all(dates=None, schemes=None) -> RunMap:
    """Every run in RESULTS_DIR (optionally only some dates/schemes), as {(date, scheme): RunOutput}."""
    dates = None if dates is None else {_as_date(d) for d in dates}
    schemes = None if schemes is None else {s.lower() for s in schemes}
    results = RunMap()
    for directory, date_obj, scheme in identify_runs():
        if (dates is None or date_obj in dates) and (schemes is None or scheme.lower() in schemes):
            results[date_obj, scheme.lower()] = _load_dir(directory, date_obj, scheme)
    return results


def load_run(date_obj, scheme: str) -> RunOutput:
    """One run, e.g. load_run("20131210", "ysu")."""
    date_obj = _as_date(date_obj)
    directory = RESULTS_DIR / f"{date_obj:%Y%m%d}_{scheme}"
    if not directory.is_dir():
        available = ", ".join(f"{d:%Y%m%d}_{s}" for _, d, s in identify_runs())
        raise FileNotFoundError(f"No run {directory.name} in {RESULTS_DIR} (available: {available})")
    return _load_dir(directory.resolve(), date_obj, scheme)


def _load_dir(directory: Path, date_obj: date, scheme: str) -> RunOutput:
    # discover stations from the .TS files present (stations need not be the same in every run)
    prefixes = sorted(p.name.split(".")[0] for p in directory.glob(f"*.{DOMAIN}.TS"))
    stations = {pfx: Station(directory, pfx) for pfx in prefixes if pfx != TOWER_PREFIX}
    tower = Tower(directory, TOWER_PREFIX)
    return RunOutput(stations=stations, tower=tower, date_obj=date_obj, scheme=scheme, directory=directory)


if __name__ == "__main__":
    # Quick tests / attempt loading all available and print metadata
    runs = load_all()
    print(f"{len(runs)} runs; dates {[f'{d:%Y%m%d}' for d in runs.dates]}; schemes {runs.schemes}")
    for run in runs.values():
        print(f"  {run}: tower at ({run.tower.latitude}, {run.tower.longitude}), {run.tower.elevation} m")
    run = next(iter(runs.values()))
    st = next(iter(run.stations.values()))
    print(st)
    print(st.df.head(3))
    print(st.ws.iloc[:3, :5])
    col = run.tower.column
    print(col.sfc.head(3))
    print(col.u.iloc[:3, :5])
