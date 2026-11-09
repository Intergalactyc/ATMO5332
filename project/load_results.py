"""
load_results.py
Load the packed WRF results (RESULTS_DIR/<event>_<scheme>/, from pack_results.py) for analysis.

Usage:
    import load_results
    runs = load_results.load_all()                  # {(event, scheme): RunOutput}
    run = load_results.load_run("20131210", "ysu")
    abern = run.stations["ABERN"]                   # tslist prefix -> Station
    abern.df[["t", "ws", "wd"]]                     # surface time series (.TS)
    abern.ws, abern.z                               # profiles: time x model level (1 = lowest)
    run.tower.column.u                              # tower_column.nc

Winds are earth-relative (WRF rotates the time series, pack_results.py the column); wd is the direction the wind
blows from. Times are naive UTC; pressures are in hPa. Station data are read on first access and then cached.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from ttu_tower.math.polar import vector_to_bearing

from prepare_runs import TOWER

RESULTS_DIR = Path(__file__).parent / "results"
DOMAIN = "d03"

# WRF's constants (share/module_model_constants.F)
R_D = 287.0  # J kg-1 K-1
CP = 7.0 * R_D / 2.0  # J kg-1 K-1
P0 = 1000.0  # hPa
XLV = 2.5e6  # J kg-1

# .TS columns (WRF README.tslist); id, id_tsloc, ix, iy are dropped
TS_COLUMNS = ["id", "hour", "id_tsloc", "ix", "iy",
              "t",       # 2 m temperature (K)
              "q",       # 2 m vapor mixing ratio (kg/kg)
              "u",       # 10 m west-east wind (m/s)
              "v",       # 10 m south-north wind (m/s)
              "psfc",    # surface pressure (hPa; Pa in the file)
              "glw",     # downward longwave at surface (W/m2)
              "gsw",     # net shortwave at surface (W/m2)
              "hfx",     # surface sensible heat flux (W/m2)
              "lh",      # surface latent heat flux (W/m2)
              "tsk",     # skin temperature (K)
              "tslb",    # top soil layer temperature (K)
              "rainc",   # accumulated cumulus precipitation (mm)
              "rainnc",  # accumulated grid-scale precipitation (mm)
              "clw"]     # column-integrated cloud water (kg/m2)

# "Abernathy   3  2 ABERN ( 33.875,-101.757) ( 195, 213) ( 33.876,-101.762) 1011.2 meters  2013-12-09_18:00:00":
# requested location, grid point, grid point location, terrain height, start
HEADER_RE = re.compile(r"^(?P<long_name>.*?)\s+\d+\s+\d+\s+\S+\s+\(.*?\)\s+\(.*?\)\s+"
                       r"\(\s*(?P<lat>[-+\d.]+)\s*,\s*(?P<lon>[-+\d.]+)\s*\)\s+(?P<elev>[-+\d.]+)\s+meters\s+(?P<start>\S+)")


def temperature(theta, p):
    """Temperature (K) from theta (K) and p (hPa), with WRF's constants."""
    return theta * (p / P0) ** (R_D / CP)


def _read_numeric(path: Path, ncols: int) -> np.ndarray:
    """A time series file's rows after its header, as an (ntimes, ncols) array."""
    with open(path) as f:
        f.readline()
        return np.array(f.read().split(), dtype=float).reshape(-1, ncols)


class Station:
    """One tslist location of a run."""

    def __init__(self, directory: Path, prefix: str):
        self.directory = Path(directory)
        self.prefix = prefix
        header = self._path("TS").open().readline()
        m = HEADER_RE.match(header)
        if m is None:
            raise ValueError(f"unrecognized header in {self._path('TS')}: {header!r}")
        self.long_name : str = m["long_name"].strip()  # e.g. "Abernathy"
        self.latitude : float = float(m["lat"])  # of the model grid point
        self.longitude : float = float(m["lon"])
        self.elevation : float = float(m["elev"])  # model terrain height, m ASL
        self.start : datetime = datetime.strptime(m["start"], "%Y-%m-%d_%H:%M:%S")

    def __repr__(self):
        return (f"{type(self).__name__}({self.prefix!r}, {self.long_name!r}, lat={self.latitude}, "
                f"lon={self.longitude}, elev={self.elevation} m)")

    def _path(self, ext: str) -> Path:
        return self.directory / f"{self.prefix}.{DOMAIN}.{ext}"

    def _times(self, hours: np.ndarray) -> pd.DatetimeIndex:
        # hours have 6 decimals, so round to the second
        return pd.DatetimeIndex(self.start + pd.to_timedelta(hours * 3600, unit="s").round("1s"), name="time")

    def _profile(self, ext: str) -> pd.DataFrame:
        with open(self._path(ext)) as f:
            f.readline()
            ncols = len(f.readline().split())
        data = _read_numeric(self._path(ext), ncols)
        levels = pd.RangeIndex(1, ncols, name="level")
        return pd.DataFrame(data[:, 1:], index=self._times(data[:, 0]), columns=levels)

    @cached_property
    def df(self) -> pd.DataFrame:
        """Surface time series (TS_COLUMNS), plus 10 m ws and wd."""
        data = _read_numeric(self._path("TS"), len(TS_COLUMNS))
        df = pd.DataFrame(data, columns=TS_COLUMNS, index=self._times(data[:, 1]))
        df = df.drop(columns=["id", "id_tsloc", "ix", "iy"])
        df["psfc"] /= 100.0
        df["ws"], df["wd"] = vector_to_bearing(df["u"], df["v"])
        return df

    @cached_property
    def gph(self) -> pd.DataFrame:
        """Geopotential height (m ASL)"""
        return self._profile("PH")

    @cached_property
    def z(self) -> pd.DataFrame:
        """Height (m AGL)"""
        return self.gph - self.elevation

    @cached_property
    def p(self) -> pd.DataFrame:
        """Pressure (hPa)"""
        return self._profile("PR") / 100.0

    @cached_property
    def qv(self) -> pd.DataFrame:
        """Water vapor mixing ratio (kg/kg)"""
        return self._profile("QV")

    @cached_property
    def theta(self) -> pd.DataFrame:
        """Potential temperature (K)"""
        return self._profile("TH")

    @cached_property
    def u(self) -> pd.DataFrame:
        """West-east wind (m/s)"""
        return self._profile("UU")

    @cached_property
    def v(self) -> pd.DataFrame:
        """South-north wind (m/s)"""
        return self._profile("VV")

    @cached_property
    def w(self) -> pd.DataFrame:
        """Vertical wind (m/s)"""
        return self._profile("WW")

    @cached_property
    def ws(self) -> pd.DataFrame:
        """Wind speed (m/s)"""
        return vector_to_bearing(self.u, self.v)[0]

    @cached_property
    def wd(self) -> pd.DataFrame:
        """Wind direction (degrees from)"""
        return vector_to_bearing(self.u, self.v)[1]


@dataclass
class TowerColumn:
    """tower_column.nc: the d03 wrfout column at the tower grid point, every 15 min. Profiles are time x level, with
    level 1 the lowest; w levels (level_w) number one more than mass levels."""
    z : pd.DataFrame  # mass level heights, m AGL
    z_w : pd.DataFrame  # w level heights, m AGL
    u : pd.DataFrame  # west-east wind, m/s
    v : pd.DataFrame  # south-north wind, m/s
    theta : pd.DataFrame  # potential temperature, K
    p : pd.DataFrame  # pressure, hPa (Pa in the file)
    qv : pd.DataFrame  # water vapor mixing ratio, kg/kg
    exch_h : pd.DataFrame  # heat exchange coefficient, m2/s (w levels; 0 at the ground)
    exch_m : pd.DataFrame  # momentum exchange coefficient, m2/s (w levels; 0 at the ground)
    sfc : pd.DataFrame  # pblh (m), ust (m/s), hfx (W/m2), lh (W/m2)
    tke_pbl : pd.DataFrame | None = None  # TKE, m2/s2, TKE schemes only: w levels, but MYNN's (QKE / 2) on mass levels


def load_tower_column(path: Path, acm2: bool = False) -> TowerColumn:
    """Read tower_column.nc. ACM2 stores the exchange coefficients of w level k + 1 at index k (module_bl_acm.F:
    EXCH_H(K) = EDDYZ(K), at ZF(K), the top of layer K), so for acm2 they are shifted up a level to match the others."""
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
    profiles["p"] /= 100.0
    if acm2:
        for name in ("exch_h", "exch_m"):
            shifted = profiles[name].shift(1, axis=1)  # the top level drops off
            shifted[1] = 0.0
            profiles[name] = shifted
    return TowerColumn(**profiles, sfc=pd.DataFrame(sfc, index=times))


class Tower(Station):
    """The tower's tslist location plus its tower_column.nc."""

    def __init__(self, directory: Path, scheme: str):
        super().__init__(directory, TOWER[1])
        self.scheme = scheme

    @cached_property
    def column(self) -> TowerColumn:
        return load_tower_column(self.directory / "tower_column.nc", acm2=self.scheme == "acm2")


@dataclass
class RunOutput:
    event : str  # "YYYYMMDD"
    scheme : str
    stations : dict[str, Station] = field(repr=False)  # tslist prefix -> Station
    tower : Tower = field(repr=False)


def identify_runs() -> list[tuple[str, str]]:
    """(event, scheme) of each run in RESULTS_DIR."""
    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(f"{RESULTS_DIR} not found")
    return sorted(tuple(d.name.split("_")) for d in RESULTS_DIR.iterdir()
                  if d.is_dir() and re.fullmatch(r"\d{8}_[^_]+", d.name))


def load_run(event: str, scheme: str) -> RunOutput:
    """One run, e.g. load_run("20131210", "ysu")."""
    directory = RESULTS_DIR / f"{event}_{scheme}"
    if not directory.is_dir():
        raise FileNotFoundError(f"no run {directory.name} in {RESULTS_DIR}")
    prefixes = sorted(p.name.split(".")[0] for p in directory.glob(f"*.{DOMAIN}.TS"))
    stations = {pfx: Station(directory, pfx) for pfx in prefixes if pfx != TOWER[1]}
    return RunOutput(event=event, scheme=scheme, stations=stations, tower=Tower(directory, scheme))


def load_all(events=None, schemes=None) -> dict[tuple[str, str], RunOutput]:
    """Every run in RESULTS_DIR, optionally only some events and schemes."""
    return {(e, s): load_run(e, s) for e, s in identify_runs()
            if (events is None or e in events) and (schemes is None or s in schemes)}


if __name__ == "__main__":
    runs = load_all()
    print(f"{len(runs)} runs")
    for run in runs.values():
        print(f"  {run}: {len(run.stations)} stations, {run.tower}")
    run = next(iter(runs.values()))
    st = next(iter(run.stations.values()))
    print(st)
    print(st.df.head(3))
    print(st.ws.iloc[:3, :5])
    col = run.tower.column
    print(col.sfc.head(3))
    print(col.u.iloc[:3, :5])
