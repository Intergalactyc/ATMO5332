"""
load_observations.py
Load an event's observations: the TTU 200 m tower (ttu-tower-processing) and the West Texas Mesonet (wtxmeso).

Usage:
    import load_observations as obs
    tower = obs.load_tower_data("20131210")       # TowerObservations
    tower.df                                      # ttu_tower's wide table, indexed by slot_start
    tower.profile("ws_mean")                      # time x boom, from the "ws_mean_b{boom}" columns
    meso = obs.load_mesonet_data("20131210")      # MesonetObservations
    abern = meso.stations["ABERN"]                # tslist prefix, as in load_results
    abern.df                                      # QC'd MESONET_COLUMNS every 15 min
    abern.native                                  # the same at the station's own 1- or 5-min cadence
    abern.raw                                     # table 1 as wtxmeso reads it, without QC
    abern.flags["stuck"]                          # wtxmeso.qc flags of a test, on raw's columns
    meso.qc_summary()                             # values each test flagged, per station and raw column

Times are naive UTC. An event's window runs from 18 UTC the day before the event date to 18 UTC on it.

Tower: the event's own ttu-tower-processing run, tagged "llj_<event>" (configs/tower/). Its 15-min slot averages are
labeled by slot start, so a slot labeled T is centred 7.5 min after the model output at T. The window holds 96 slots.

Mesonet: MESONET_DIR/<start>_<end>/ (YYYYmmddHHMM), one folder per event, for the tslist's stations
(prepare_runs.select_stations); those without data are listed in MesonetObservations.missing. QC (wtxmeso.qc): values
failing the range or stuck tests are masked, as are raw columns with less than wtxmeso.qc.MIN_COVERAGE valid
(MesonetStation.excluded); the consistency test only flags. The value at T averages the records within 7.5 min of T,
so it is centred on the model output, and is NaN with fewer than half the expected records. u and v are vector means,
and wd is recomputed from them.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import ttu_tower
from ttu_tower import constants as tower_constants
from ttu_tower.io import store
from ttu_tower.math.polar import bearing_to_vector, vector_to_bearing
from ttu_tower.physics import thermo
import wtxmeso
from wtxmeso import qc
from wtxmeso.station import Station as WtxStation

import prepare_runs
from load_results import INTERVAL

TOWER_HEIGHTS: dict[int, float] = tower_constants.HEIGHTS  # boom -> m AGL
MESONET_DIR = Path(os.environ.get("MESONET_DIR", Path.home() / "Data" / "Mesonet_LLJs"))

# Standardized mesonet columns, named as in load_results' Station.df
MESONET_COLUMNS = ["t",     # 2 m temperature (K)
                   "q",     # 1.5 m water vapor mixing ratio (kg/kg)
                   "psfc",  # station pressure (Pa)
                   "u",     # 10 m west-east wind (m/s)
                   "v",     # 10 m south-north wind (m/s)
                   "ws",    # 10 m scalar-mean wind speed (m/s)
                   "wd"]    # 10 m wind direction (degrees from)


def event_window(event: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end) of an event's window."""
    start = pd.to_datetime(event, format="%Y%m%d") + prepare_runs.INIT_OFFSET
    return start, start + pd.Timedelta(hours=24)


# ---------------------------------------------------------------------------------------------------------------
# Tower
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class TowerObservations:
    event : str
    start : pd.Timestamp
    end : pd.Timestamp
    df : pd.DataFrame = field(repr=False)  # wide table, one row per slot, indexed by slot_start

    def profile(self, var: str) -> pd.DataFrame:
        """time x boom frame of the "{var}_b{boom}" columns. A variant suffix goes after the boom: "tke_mrd" gives
        the tke_b{boom}_mrd columns."""
        cols = {boom: self.df[c] for boom in TOWER_HEIGHTS if (c := _with_variant(var, boom)) in self.df.columns}
        if not cols:
            raise KeyError(f"no tower columns for {var!r} (expected e.g. {_with_variant(var, 4)})")
        return pd.DataFrame(cols).rename_axis(columns="boom")


def _with_variant(var: str, boom: int) -> str:
    for variant in ("_mrd_unexcised", "_mrd", "_naive"):
        if var.endswith(variant):
            return f"{var[:-len(variant)]}_b{boom}{variant}"
    return f"{var}_b{boom}"


def load_tower_data(event: str, columns: list[str] | None = None) -> TowerObservations:
    """The tower's slots in the event window. columns limits the wide-table columns read."""
    start, end = event_window(event)
    wide_dir = ttu_tower.find_run(f"llj_{event}") / "tertiary" / "wide"
    paths = sorted(wide_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no tower wide files in {wide_dir}")
    if columns is not None:
        columns = ["slot_start"] + [c for c in columns if c != "slot_start"]
    df = store.read_paths(paths, columns=columns).set_index("slot_start").sort_index()
    df.index = df.index.tz_convert("UTC").tz_localize(None)
    return TowerObservations(event=event, start=start, end=end, df=df.loc[start:end - INTERVAL])


# ---------------------------------------------------------------------------------------------------------------
# Mesonet
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class MesonetStation:
    prefix : str  # tslist prefix
    station : WtxStation = field(repr=False)  # wtxmeso metadata
    raw : pd.DataFrame = field(repr=False)  # table 1 as read by wtxmeso: source columns and units, no QC
    flags : dict[str, pd.DataFrame] = field(repr=False)  # wtxmeso.qc test -> flags (True = failed), raw columns
    coverage : pd.Series = field(repr=False)  # raw column -> fraction valid after QC
    excluded : list[str]  # raw columns masked for coverage below wtxmeso.qc.MIN_COVERAGE
    native : pd.DataFrame = field(repr=False)  # QC'd MESONET_COLUMNS at the native cadence
    df : pd.DataFrame = field(repr=False)  # native resampled to INTERVAL

    @property
    def id(self) -> str:
        return self.station.id

    @property
    def latitude(self) -> float:
        return self.station.latitude

    @property
    def longitude(self) -> float:
        return self.station.longitude

    @property
    def elevation(self) -> float:
        """m ASL"""
        return self.station.elevation * 0.3048

    def __repr__(self):
        return (f"MesonetStation({self.prefix!r}, {self.id!r}, {self.station.name!r}, lat={self.latitude}, "
                f"lon={self.longitude}, {len(self.native)} records" +
                (f", excluded {self.excluded}" if self.excluded else "") + ")")


@dataclass
class MesonetObservations:
    event : str
    start : pd.Timestamp
    end : pd.Timestamp
    stations : dict[str, MesonetStation] = field(repr=False)  # tslist prefix -> MesonetStation
    missing : list[str] = field(default_factory=list)  # prefixes of stations without data

    def __repr__(self):
        return (f"MesonetObservations({self.event}, {len(self.stations)} stations"
                + (f", missing {self.missing}" if self.missing else "") + ")")

    def qc_summary(self) -> pd.DataFrame:
        """Values each QC test flagged, indexed by (station prefix, test), one column per raw column."""
        return pd.concat({pfx: qc.summary(s.flags) for pfx, s in self.stations.items()},
                         names=["station", "test"]).fillna(0).astype(int)


def _standardize(raw: pd.DataFrame) -> pd.DataFrame:
    """Table 1 (wtxmeso names, source units) -> MESONET_COLUMNS."""
    df = pd.DataFrame(index=raw.index)
    df["t"] = raw["2m Temp"] + 273.15
    df["psfc"] = raw["barometric pressure"] * 100.0
    e = thermo.water_partial_pressure(raw["1.5m RH"] / 100.0,
                                      thermo.saturation_vapor_pressure(raw["1.5m Temp"] + 273.15))  # kPa
    df["q"] = thermo.water_air_mixing_ratio(e, df["psfc"] / 1000.0)
    df["u"], df["v"] = bearing_to_vector(raw["10m WS vector"].to_numpy(), raw["10m WD"].to_numpy())
    df["ws"] = raw["10m WS scalar"]
    df["wd"] = raw["10m WD"]
    return df[MESONET_COLUMNS]


def _resample(native: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """native -> every INTERVAL from start to end (see the module docstring)."""
    step = native.index.to_series().diff().median()
    needed = max(1, int(np.ceil(0.5 * (INTERVAL / step))))
    groups = native.groupby((native.index + INTERVAL / 2).floor(INTERVAL))
    out = groups.mean().where(groups.count() >= needed)
    _, out["wd"] = vector_to_bearing(out["u"].to_numpy(), out["v"].to_numpy())
    return out.reindex(pd.date_range(start, end, freq=INTERVAL, name="time"))[MESONET_COLUMNS]


def load_mesonet_data(event: str, data_dir: Path = MESONET_DIR) -> MesonetObservations:
    """Table 1 (meteorological) data of the tslist's mesonet stations in the event window."""
    start, end = event_window(event)
    window = f"{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}"
    event_dir = Path(data_dir) / window
    if not event_dir.is_dir():
        raise FileNotFoundError(f"no mesonet data for {event}: {event_dir} not found")

    selected = {name: pfx for name, pfx, _, _ in prepare_runs.select_stations()}
    table = (pd.read_csv(prepare_runs.STATIONS).set_index("archive_name", drop=False).loc[list(selected)]
             .assign(ID=lambda t: t["ID"].fillna(pd.Series(selected))))  # prefix where there is no ID (Hobbs, Lubbock)
    reader = wtxmeso.Reader(table, "atmospheric")

    stations, missing = {}, []
    for name, pfx in selected.items():
        path = event_dir / f"{name}1_{window}.txt"
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(pfx)
            continue
        raw = reader.load_file(path)
        raw.index = raw.index.tz_convert("UTC").tz_localize(None).rename("time")
        raw = raw.sort_index().loc[start:end]
        if raw.empty:
            missing.append(pfx)
            continue
        flags = qc.flag(raw)
        clean = qc.apply(raw, flags)
        coverage = qc.coverage(clean, start, end)
        # columns of sensors the station lacks are all NaN anyway
        excluded = [c for c in coverage.index[coverage < qc.MIN_COVERAGE] if raw[c].notna().any()]
        clean[excluded] = np.nan
        native = _standardize(clean)
        stations[pfx] = MesonetStation(prefix=pfx, station=reader.station(name), raw=raw, flags=flags,
                                       coverage=coverage, excluded=excluded, native=native,
                                       df=_resample(native, start, end))
    return MesonetObservations(event=event, start=start, end=end, stations=stations, missing=sorted(missing))


if __name__ == "__main__":
    for event in prepare_runs.EVENTS:
        print(f"--- {event}: {' -> '.join(map(str, event_window(event)))}")
        try:
            tower = load_tower_data(event)
            print(f"  {tower}: {len(tower.df)} slots, {tower.df.index[0]} -> {tower.df.index[-1]}")
            meso = load_mesonet_data(event)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  {e}")
            continue
        print(f"  {meso}")
        st = next(iter(meso.stations.values()))
        print(f"  {st}: {st.df.index[0]} -> {st.df.index[-1]}")
