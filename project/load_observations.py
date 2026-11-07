"""
load_observations.py
Module for loading the observations the runs are compared against, for one event: the TTU 200 m tower
(ttu-tower-processing) and the West Texas Mesonet stations (wtxmeso)

Usage:
    import load_observations as obs
    tower = obs.load_tower_data("20131210")       # TowerObservations; dates as in load_results ("YYYYMMDD", date, ...)
    tower.df                                      # ttu_tower's wide table (15-min slots), indexed by slot_start (UTC)
    tower.profile("ws_mean")                      # time x boom, from the "{var}_b{boom}" columns
    meso = obs.load_mesonet_data("20131210")      # MesonetObservations
    abern = meso.stations["ABERN"]                # same prefixes as load_results' RunOutput.stations
    abern.df[["t", "q", "u", "v", "ws", "wd"]]    # QC'd standardized SI columns every 15 min (see MESONET_COLUMNS)
    abern.native                                  # the same at the station's own 1- or 5-min cadence
    abern.raw                                     # table 1 as wtxmeso reads it (source names and units, no QC)
    abern.flags["stuck"]                          # wtxmeso.qc flags of each test (on raw's columns)
    meso.qc_summary()                             # values each test flagged, per station and raw column

Times are naive UTC everywhere (start, end and every index), the same as load_results, so the frames line up with
the model output directly.

Time period: each event's window is 18 UTC the day before the event date to 18 UTC on it, the run period
(prepare_runs.INIT_OFFSET). Mesonet records are read at their own timestamps in [start, end], both ends included
(1-min or 5-min cadence, depending on the station). Tower records are slot averages (15 min for the per-event runs)
labeled by the start of their averaging slot, and only slots lying entirely in the window are kept: slot_start from
start to end - slot (96 slots). A slot labeled T covers [T, T + 15 min), so its center is T + 7.5 min, half a slot
after the model's instantaneous output at T (the mesonet values, by contrast, are centered on T).

Mesonet QC (wtxmeso.qc): values failing the range or stuck sensor tests are set to NaN, and so is every value of a raw
column with less than wtxmeso.qc.MIN_COVERAGE (75 %) valid records in the window (listed in MesonetStation.excluded).
The consistency test only flags. All tests' flags are kept in MesonetStation.flags.

Mesonet resampling (MESONET_INTERVAL, 15 min = the d03 history interval): each value at time T aggregates the records
stamped within 7.5 min of T (T-5, T, T+5 for a 5-min station; T-7 ... T+7 for a 1-min one), so it is centered on
the model's instantaneous output time. Means for most columns; vector means for u, v (with ws_vec and wd recomputed
from them); maximum for gust; total for precip; RMS for the standard deviations. A value needs at least half the
records its interval should hold, else it is NaN. The index runs from start to end every 15 min (97 times).

Stations: the mesonet stations are the ones prepare_runs puts in the tslist (the official location list,
meta/wtm_2013_station_locations.csv, within prepare_runs.BOUNDS), keyed by the same tslist prefix. A station
without data for the event is listed in MesonetObservations.missing instead.

Data locations:
    tower       the event's own ttu-tower-processing run, registered as tower_tag(event) = "llj_YYYYMMDD" (its
                tertiary wide export), made from configs/tower/llj_YYYYMMDD.toml at 15-min slots
    mesonet     MESONET_DIR/<YYYYmmddHHMM>_<YYYYmmddHHMM>/<archive_name>1_<window>.txt, one folder per event
                window, with the download_summary_<window>.csv files giving each station's 4-letter ID (the tslist
                prefix stands in for a station none of them gives an ID, e.g. Hobbs and Lubbock)
"""

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_dataset
import pyarrow.parquet as pq

import ttu_tower
from ttu_tower import constants as tower_constants
from ttu_tower.config import config_from_dict
from ttu_tower.io import store
from ttu_tower.math.polar import bearing_to_vector, vector_to_bearing
from ttu_tower.physics import thermo
from ttu_tower.timegrid import Cadence
import wtxmeso
from wtxmeso import qc
from wtxmeso.station import Station as WtxStation

import prepare_runs

TOWER_TAG_FORMAT = "llj_{:%Y%m%d}"  # each event's own ttu-tower-processing run (configs/tower/<tag>.toml)
TOWER_HEIGHTS: dict[int, float] = tower_constants.HEIGHTS  # boom -> height, m AGL

MESONET_DIR = Path(os.environ.get("MESONET_DIR", Path.home() / "Data" / "Mesonet_LLJs"))
MESONET_INTERVAL = pd.Timedelta("15min")  # resampling interval (d03 history_interval)


def _as_date(d) -> date:
    """Accept date, datetime, or "YYYYMMDD" / "YYYY-MM-DD" strings (as load_results does)."""
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d).replace("-", ""), "%Y%m%d").date()


def event_window(event) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end) of an event's analysis period: 18 UTC the day before to 18 UTC on the event date (naive UTC)."""
    start = pd.Timestamp(_as_date(event)) + prepare_runs.INIT_OFFSET
    return start, start + timedelta(hours=24)


# ---------------------------------------------------------------------------------------------------------------
# Tower
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class TowerObservations:
    date_obj : date
    start : pd.Timestamp  # window start (UTC)
    end : pd.Timestamp  # window end (UTC)
    df : pd.DataFrame = field(repr=False)  # ttu_tower wide table, one row per slot, indexed by slot_start (UTC)
    slot : pd.Timedelta = pd.Timedelta("15min")  # averaging period of a record (the run's [cadence].slot_min)
    tag : str = ""  # the ttu-tower-processing run read
    heights : dict[int, float] = field(default_factory=lambda: dict(TOWER_HEIGHTS), repr=False)  # boom -> m AGL

    def profile(self, var: str) -> pd.DataFrame:
        """time x boom frame of the wide table's "{var}_b{boom}" columns, e.g. profile("ws_mean") or
        profile("tke_mrd") for tke_b{boom}_mrd ("{var}" may end in a variant: "_mrd", "_naive", ...)."""
        cols = {boom: self.df[c] for boom in self.heights if (c := _with_variant(var, boom)) in self.df.columns}
        if not cols:
            raise KeyError(f"no tower columns for {var!r} (expected e.g. {var}_b4)")
        return pd.DataFrame(cols).rename_axis(columns="boom")


def _with_variant(var: str, boom: int) -> str:
    """Wide-table column of var at boom: variant suffixes go after the boom ("tke_mrd" -> "tke_b4_mrd")."""
    for variant in ("_mrd_unexcised", "_mrd", "_naive"):
        if var.endswith(variant):
            return f"{var[:-len(variant)]}_b{boom}{variant}"
    return f"{var}_b{boom}"


def tower_tag(event) -> str:
    """The tag of an event's own ttu-tower-processing run, e.g. "llj_20131210"."""
    return TOWER_TAG_FORMAT.format(_as_date(event))


def tower_slot(run_dir: Path) -> pd.Timedelta:
    """A run's slot length, from the config its tertiary stage recorded (10 min for runs without [cadence])."""
    with open(Path(run_dir) / "tertiary" / "config.resolved.json") as f:
        cfg = config_from_dict(json.load(f))
    return pd.Timedelta(minutes=Cadence.of(cfg).slot_min)


def load_tower_data(event, columns: list[str] | None = None, tag: str | None = None) -> TowerObservations:
    """The tower's slot records in the event window, from the event's own ttu-tower-processing run (tower_tag(event),
    15-min slots), or from the run registered as `tag` if given (e.g. "oneyear", 10-min slots).
    columns limits which wide-table columns are read (slot_start is always included)."""
    start, end = event_window(event)
    tag = tag or tower_tag(event)
    run_dir = ttu_tower.find_run(tag)
    slot = tower_slot(run_dir)
    wide_dir = run_dir / "tertiary" / "wide"
    # wide files are per calendar month in the run's output timezone; a day of margin covers any offset from UTC
    months = sorted({t.strftime("%Y-%m") for t in (start - timedelta(days=1), end + timedelta(days=1))})
    paths = [p for m in months if (p := wide_dir / f"{m}.parquet").is_file()]
    if not paths:
        raise FileNotFoundError(f"no tower wide files for {months} in {wide_dir}")
    if columns is not None:
        columns = ["slot_start"] + [c for c in columns if c != "slot_start"]
    # pyarrow only compares timestamps of the same type (incl. time zone), so give the bounds slot_start's type
    slot_type = pq.read_schema(paths[0]).field("slot_start").type
    lo, hi = (pa.scalar(t.tz_localize("UTC"), type=slot_type) for t in (start, end - slot))
    slot_start = pa_dataset.field("slot_start")
    df = store.read_paths(paths, columns=columns, filter=(slot_start >= lo) & (slot_start <= hi))
    df = df.set_index("slot_start").sort_index()
    df.index = df.index.tz_convert("UTC").tz_localize(None)  # naive UTC, like load_results
    return TowerObservations(date_obj=_as_date(event), start=start, end=end, df=df, slot=slot, tag=tag)


# ---------------------------------------------------------------------------------------------------------------
# Mesonet
# ---------------------------------------------------------------------------------------------------------------

# Standardized columns of MesonetStation.df and .native: name -> (description, units). Names match load_results'
# Station.df where there is a model counterpart (t, q, u, v, psfc, ws, wd). Descriptions are of the native records;
# see the module docstring for how each is aggregated to 15 min.
MESONET_COLUMNS = {
    "t":       ("2 m temperature", "K"),
    "t_1p5":   ("1.5 m temperature", "K"),
    "t_9":     ("9 m temperature", "K"),
    "td":      ("1.5 m dewpoint", "K"),
    "rh":      ("1.5 m relative humidity", "fraction"),
    "q":       ("1.5 m water vapor mixing ratio, from rh, t_1p5 and psfc", "kg/kg"),
    "psfc":    ("station pressure", "Pa"),
    "ws":      ("10 m scalar-mean wind speed", "m/s"),
    "ws_vec":  ("10 m vector-mean wind speed", "m/s"),
    "wd":      ("10 m wind direction (from)", "degrees"),
    "u":       ("10 m west-east wind, vector mean (from ws_vec, wd)", "m/s"),
    "v":       ("10 m south-north wind, vector mean (from ws_vec, wd)", "m/s"),
    "wd_std":  ("10 m wind direction standard deviation", "degrees"),
    "ws_std":  ("10 m wind speed standard deviation", "m/s"),
    "gust":    ("10 m peak 3-s gust", "m/s"),
    "ws_2m":   ("2 m wind speed", "m/s"),
    "ws_20ft": ("20 ft (6.1 m) wind speed", "m/s"),
    "solar":   ("downward solar radiation (the model's gsw is net shortwave)", "W/m2"),
    "precip":  ("precipitation during the record's interval (not accumulated like rainc/rainnc)", "mm"),
}


@dataclass
class MesonetStation:
    prefix : str  # tslist prefix, e.g. "ABERN"
    station : WtxStation = field(repr=False)  # wtxmeso metadata: name, id, latitude, longitude, elevation (ft)
    raw : pd.DataFrame = field(repr=False)  # table 1 as read by wtxmeso (source columns and units, missing -> NaN)
    flags : dict[str, pd.DataFrame] = field(repr=False)  # wtxmeso.qc test -> its flags (True = failed), raw columns
    coverage : pd.Series = field(repr=False)  # raw column -> fraction of the window's records valid after QC masking
    excluded : list[str]  # raw columns with data, masked entirely for coverage below wtxmeso.qc.MIN_COVERAGE
    native : pd.DataFrame = field(repr=False)  # QC'd standardized SI columns (MESONET_COLUMNS), native cadence
    df : pd.DataFrame = field(repr=False)  # native resampled to MESONET_INTERVAL, indexed by time (UTC)

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
        """Station elevation, m ASL."""
        return self.station.elevation * 0.3048

    def __repr__(self):
        return (f"MesonetStation({self.prefix!r}, {self.id!r}, {self.station.name!r}, lat={self.latitude}, "
                f"lon={self.longitude}, {len(self.native)} records" +
                (f", excluded {self.excluded}" if self.excluded else "") + ")")


@dataclass
class MesonetObservations:
    date_obj : date
    start : pd.Timestamp  # window start (UTC)
    end : pd.Timestamp  # window end (UTC)
    stations : dict[str, MesonetStation] = field(repr=False)  # tslist prefix -> MesonetStation
    missing : list[str] = field(default_factory=list)  # selected stations (prefixes) without data for the event
    units : dict[str, str] = field(default_factory=dict, repr=False)  # raw column -> units, from wtxmeso

    def __repr__(self):
        return (f"MesonetObservations({self.date_obj:%Y%m%d}, {len(self.stations)} stations"
                + (f", missing {self.missing}" if self.missing else "") + ")")

    def qc_summary(self) -> pd.DataFrame:
        """Values each QC test flagged, indexed by (station prefix, test), one column per raw column."""
        return pd.concat({pfx: qc.summary(s.flags) for pfx, s in self.stations.items()},
                         names=["station", "test"]).fillna(0).astype(int)


def selected_stations() -> list[tuple[str, str, float, float]]:
    """(archive_name, prefix, lat, lon) of the mesonet stations prepare_runs puts in the tslist."""
    lat_min, lat_max, lon_min, lon_max = prepare_runs.BOUNDS
    return [s for s in prepare_runs.read_stations(prepare_runs.STATIONS)
            if lat_min <= s[2] <= lat_max and lon_min <= s[3] <= lon_max]


def _station_ids(data_dir: Path, window: str) -> dict[str, str]:
    """archive_name -> 4-letter station ID (e.g. "ODON3"), from the window's download summary, with any other
    window's summaries filling in stations it lacks (some folders, e.g. 20140125's, have no summary)."""
    own = data_dir / window / f"download_summary_{window}.csv"
    others = sorted(p for p in data_dir.glob("*/download_summary_*.csv") if p != own)
    ids = {}
    for path in ([own] if own.is_file() else []) + others:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["table"] == "1" and row["ID"]:  # some stations (Hobbs, Lubbock) have no ID in any summary
                    ids.setdefault(row["archive_name"], row["ID"])
    return ids


def _standardize(raw: pd.DataFrame) -> pd.DataFrame:
    """Table 1 columns (wtxmeso names, source units) -> MESONET_COLUMNS."""
    df = pd.DataFrame(index=raw.index)
    df["t"] = raw["2m Temp"] + 273.15
    df["t_1p5"] = raw["1.5m Temp"] + 273.15
    df["t_9"] = raw["9m Temp"] + 273.15
    df["td"] = raw["1.5m Dewpoint"] + 273.15
    df["rh"] = raw["1.5m RH"] / 100.0
    df["psfc"] = raw["barometric pressure"] * 100.0
    e = thermo.water_partial_pressure(df["rh"], thermo.saturation_vapor_pressure(df["t_1p5"]))  # kPa
    df["q"] = thermo.water_air_mixing_ratio(e, df["psfc"] / 1000.0)
    df["ws"] = raw["10m WS scalar"]
    df["ws_vec"] = raw["10m WS vector"]
    df["wd"] = raw["10m WD"]
    df["u"], df["v"] = bearing_to_vector(df["ws_vec"].to_numpy(), df["wd"].to_numpy())
    df["wd_std"] = raw["WD Standard Deviation"]
    df["ws_std"] = raw["WS Standard Deviation"]
    df["gust"] = raw["10m Peak Gust"]
    df["ws_2m"] = raw["2m WS"]
    df["ws_20ft"] = raw["20ft WS"]
    df["solar"] = raw["2m Solar Radiation"]
    df["precip"] = raw["Precip"] * 25.4
    return df[list(MESONET_COLUMNS)]


def _resample(native: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
              interval: pd.Timedelta = MESONET_INTERVAL) -> pd.DataFrame:
    """Standardized native records -> one value per `interval` from start to end, each from the records stamped within
    interval / 2 of it (see the module docstring), NaN where fewer than half the expected records are valid."""
    step = native.index.to_series().diff().median()
    needed = max(1, int(np.ceil(0.5 * (interval / step))))
    keys = (native.index + interval / 2).floor(interval)
    groups = native.groupby(keys)
    out = groups.mean()
    out["gust"] = groups["gust"].max()
    out["precip"] = groups["precip"].sum()
    stds = ["wd_std", "ws_std"]
    out[stds] = np.sqrt((native[stds] ** 2).groupby(keys).mean())
    out = out.where(groups.count() >= needed)
    out["ws_vec"], out["wd"] = vector_to_bearing(out["u"].to_numpy(), out["v"].to_numpy())
    out = out.reindex(pd.date_range(start, end, freq=interval, name="time"))
    return out[list(MESONET_COLUMNS)]


def load_mesonet_data(event, data_dir: Path = MESONET_DIR, interval=MESONET_INTERVAL) -> MesonetObservations:
    """Table 1 (meteorological) data of the selected mesonet stations in the event window, QC'd and resampled to
    `interval` (None: df is the native records)."""
    start, end = event_window(event)
    window = f"{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}"
    event_dir = Path(data_dir) / window
    if not event_dir.is_dir():
        raise FileNotFoundError(f"no mesonet data for {_as_date(event):%Y%m%d}: {event_dir} not found")

    # station metadata for wtxmeso: the location list's rows for the selected stations, plus their IDs (the tslist
    # prefix for a station the summaries give no ID)
    ids = _station_ids(Path(data_dir), window)
    selected = [(name, pfx) for name, pfx, _, _ in selected_stations()]
    locations = pd.read_csv(prepare_runs.STATIONS).set_index("archive_name", drop=False)
    table = locations.loc[[name for name, _ in selected]].assign(ID=[ids.get(name, pfx) for name, pfx in selected])
    reader = wtxmeso.Reader(table, "atmospheric")

    stations, missing = {}, []
    for name, pfx in selected:
        path = event_dir / f"{name}1_{window}.txt"
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(pfx)
            continue
        raw = reader.load_file(path)  # station inferred from the file name's archive name
        raw.index = raw.index.tz_convert("UTC").tz_localize(None).rename("time")  # naive UTC, like load_results
        raw = raw.sort_index().loc[start:end]
        if raw.empty:
            missing.append(pfx)
            continue
        flags = qc.flag(raw)
        clean = qc.apply(raw, flags)  # range and stuck; consistency only flags
        coverage = qc.coverage(clean, start, end)
        # columns with data but too little of it (a sensor the station lacks, e.g. 20 ft wind, is all NaN anyway)
        excluded = [c for c in coverage.index[coverage < qc.MIN_COVERAGE] if raw[c].notna().any()]
        clean[excluded] = np.nan
        native = _standardize(clean)
        df = native if interval is None else _resample(native, start, end, pd.Timedelta(interval))
        stations[pfx] = MesonetStation(prefix=pfx, station=reader.station(name), raw=raw, flags=flags,
                                       coverage=coverage, excluded=excluded, native=native, df=df)
    return MesonetObservations(date_obj=_as_date(event), start=start, end=end, stations=stations,
                               missing=sorted(missing), units=dict(reader.units))


if __name__ == "__main__":
    for event in prepare_runs.EVENTS:
        print(f"--- {event}: {event_window(event)[0]} -> {event_window(event)[1]}")
        tower = load_tower_data(event)
        print(f"  {tower}: {len(tower.df)} slots, {tower.df.index[0]} -> {tower.df.index[-1]}")
        try:
            meso = load_mesonet_data(event)
        except FileNotFoundError as e:
            print(f"  {e}")
            continue
        print(f"  {meso}")
        st = next(iter(meso.stations.values()))
        print(f"  {st}: {st.df.index[0]} -> {st.df.index[-1]}")
