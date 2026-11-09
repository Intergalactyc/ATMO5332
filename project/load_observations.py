"""
load_observations.py
Load an event's observations: the TTU 200 m tower (ttu-tower-processing) and the West Texas Mesonet (wtxmeso).

Usage:
    import load_observations as obs
    tower = obs.load_tower_data("20131210")       # structs.TowerData, as transform_tower.get_tower_model's
    tower["ws"]                                   # slot x boom
    tower.at("2013-12-10 06:00")                  # boom x [height, *variables], for a profile
    meso = obs.load_mesonet_data("20131210")      # MesonetObservations
    abern = meso.stations["ABERN"]                # tslist prefix, as in load_results
    abern.df                                      # QC'd MESONET_COLUMNS every 15 min
    abern.native                                  # the same at the station's own 1- or 5-min cadence
    abern.raw                                     # table 1 as wtxmeso reads it, without QC
    abern.flags["stuck"]                          # wtxmeso.qc flags of a test, on raw's columns
    meso.qc_summary()                             # values each test flagged, per station and raw column

Times are naive UTC; pressures are in hPa. An event's window runs from 18 UTC the day before the event date to 18 UTC
on it.

Tower: the event's own ttu-tower-processing run, tagged "llj_<event>" (configs/tower/), reduced to the variables with
model counterparts. Its 15-min slot averages are labeled by slot start, so a slot labeled T is centred
7.5 min after the model output at T. The window holds 96 slots; missing ones are NaN.

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
from ttu_tower.io import store
from ttu_tower.math.polar import bearing_to_vector, vector_to_bearing
from ttu_tower.physics import thermo
import wtxmeso
from wtxmeso import qc
from wtxmeso.station import Station as WtxStation

import prepare_runs
from structs import (TOWER_HEIGHTS, TOWER_VARIABLES, TowerData, centred_means, event_window, mesonet_times,
                     tower_slots)

# Observed TOWER_VARIABLES: ttu_tower wide-table column of boom b, flux variant v. heat_flux has none (the sonic gives
# <w'theta_v'>); load_tower_data adds momt_flux = ustar^2. Slow sensors give theta (pt), t, vpt, p and qv (r);
# exch_h = -<w'theta_v'> / dvpt/dz and exch_m is shear production / shear^2; rig is of ttu_tower's first stencil,
# log_linear (ttu_tower output-schema.md).
TOWER_OBSERVED = {"u": "ue_mean_b{b}", "v": "vn_mean_b{b}", "ws": "ws_vector_mean_b{b}", "ws_mean": "ws_mean_b{b}",
                  "wd": "wd_mean_b{b}", "theta": "pt_b{b}", "t": "t_b{b}", "vpt": "vpt_b{b}", "p": "p_b{b}",
                  "qv": "r_b{b}", "exch_h": "kh_b{b}_{v}", "exch_m": "km_b{b}_{v}", "tke": "tke_b{b}_{v}",
                  "buoy_flux": "wvpts_cov_b{b}_{v}", "ustar": "ustar_b{b}_{v}",
                  "obukhov_length": "obukhov_length_b{b}_{v}", "rig": "rig_b{b}"}

MESONET_DIR = Path(os.environ.get("MESONET_DIR", Path.home() / "Data" / "Mesonet_LLJs"))

# Standardized mesonet columns, named as in load_results' Station.df
MESONET_COLUMNS = ["t",     # 2 m temperature (K)
                   "q",     # 1.5 m water vapor mixing ratio (kg/kg)
                   "psfc",  # station pressure (hPa)
                   "u",     # 10 m west-east wind (m/s)
                   "v",     # 10 m south-north wind (m/s)
                   "ws",    # 10 m scalar-mean wind speed (m/s)
                   "wd"]    # 10 m wind direction (degrees from)


# ---------------------------------------------------------------------------------------------------------------
# Tower
# ---------------------------------------------------------------------------------------------------------------

def load_tower_data(event: str, variant: str = "mrd") -> TowerData:
    """The observed tower: TOWER_OBSERVED at the booms of TOWER_HEIGHTS, with the turbulence quantities (exch_h ...
    ustar, obukhov_length) of ttu_tower's variant "mrd", "mrd_unexcised" or "naive"."""
    wide_dir = ttu_tower.find_run(f"llj_{event}") / "tertiary" / "wide"
    paths = sorted(wide_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no tower wide files in {wide_dir}")
    columns = {var: [col.format(b=b, v=variant) for b in TOWER_HEIGHTS] for var, col in TOWER_OBSERVED.items()}
    df = store.read_paths(paths, columns=["slot_start", *sum(columns.values(), [])]).set_index("slot_start")
    df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df.reindex(tower_slots(event))
    booms = pd.Index(list(TOWER_HEIGHTS), name="boom")
    profiles = {var: df[cols].set_axis(booms, axis=1) for var, cols in columns.items()}
    profiles["p"] *= 10.0  # kPa -> hPa
    profiles["momt_flux"] = profiles["ustar"] ** 2
    return TowerData(source="obs", event=event, profiles={var: profiles[var] for var in TOWER_VARIABLES
                                                          if var in profiles}, sfc=pd.DataFrame(index=df.index))


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
    df : pd.DataFrame = field(repr=False)  # native at structs.mesonet_times

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
    df["psfc"] = raw["barometric pressure"]
    e = thermo.water_partial_pressure(raw["1.5m RH"] / 100.0,
                                      thermo.saturation_vapor_pressure(raw["1.5m Temp"] + 273.15))  # kPa
    df["q"] = thermo.water_air_mixing_ratio(e, df["psfc"] / 10.0)
    df["u"], df["v"] = bearing_to_vector(raw["10m WS vector"].to_numpy(), raw["10m WD"].to_numpy())
    df["ws"] = raw["10m WS scalar"]
    df["wd"] = raw["10m WD"]
    return df[MESONET_COLUMNS]


def _resample(native: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    """native -> times (see the module docstring)."""
    out = centred_means(native, times)
    _, out["wd"] = vector_to_bearing(out["u"].to_numpy(), out["v"].to_numpy())
    return out[MESONET_COLUMNS]


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
                                       df=_resample(native, mesonet_times(event)))
    return MesonetObservations(event=event, start=start, end=end, stations=stations, missing=sorted(missing))


if __name__ == "__main__":
    for event in prepare_runs.EVENTS:
        print(f"--- {event}: {' -> '.join(map(str, event_window(event)))}")
        try:
            tower = load_tower_data(event)
            print(f"  {tower}: {tower['ws'].notna().any(axis=1).sum()} slots with winds")
            meso = load_mesonet_data(event)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  {e}")
            continue
        print(f"  {meso}")
        st = next(iter(meso.stations.values()))
        print(f"  {st}: {st.df.index[0]} -> {st.df.index[-1]}")
