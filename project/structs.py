"""
structs.py
The data structures the observations and the model output share, and the variables and time grids they are on:
    TowerData       the 200 m tower: load_observations.load_tower_data, transform_tower.get_tower_model
    MesonetPoints   the mesonet stations: transform_meso.get_mesonet_observations, transform_meso.get_mesonet_model

Times are naive UTC; pressures are in hPa. An event's window runs from 18 UTC the day before the event date to 18 UTC
on it. The tower has 96 slots [T, T + INTERVAL), labeled by their start T (tower_slots); the mesonet has 97 times T,
end included, each the mean of the samples within INTERVAL / 2 of T (mesonet_times, centred_means).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ttu_tower import constants as tower_constants

import prepare_runs

INTERVAL = pd.Timedelta("15min")  # d03 history_interval, and the tower's slot length

TOWER_HEIGHTS: dict[int, float] = tower_constants.HEIGHTS  # boom -> m AGL

# Tower profile variables, named as in load_results' TowerColumn
TOWER_VARIABLES = ["u",          # west-east wind (m/s)
                   "v",          # south-north wind (m/s)
                   "ws",         # speed of the mean wind vector (m/s)
                   "ws_mean",    # scalar-mean wind speed (m/s)
                   "wd",         # direction of the mean wind vector (degrees from)
                   "theta",      # potential temperature (K)
                   "t",          # temperature (K)
                   "vpt",        # virtual potential temperature (K)
                   "p",          # pressure (hPa)
                   "qv",         # water vapor mixing ratio (kg/kg)
                   "exch_h",     # heat exchange coefficient (m2/s)
                   "exch_m",     # momentum exchange coefficient (m2/s)
                   "tke",        # turbulent kinetic energy (m2/s2)
                   "heat_flux",  # <w'theta'> (K m/s)
                   "buoy_flux",  # <w'theta_v'> (K m/s)
                   "momt_flux",  # |<u'w'>, <v'w'>| (m2/s2)
                   "ustar",      # friction velocity (m/s)
                   "obukhov_length",  # local Obukhov length -ustar^3 vpt / (kappa g buoy_flux) (m)
                   "rig"]        # gradient Richardson number N^2 / shear^2

# Mesonet variables, named as in load_results' Station.df except t2 (its t)
MESONET_VARIABLES = ["t2",    # 2 m temperature (K)
                     "t9",    # 9 m temperature (K)
                     "q",     # water vapor mixing ratio (kg/kg): 2 m in the model, 1.5 m in the observations
                     "td",    # dewpoint (K), from q and psfc
                     "psfc",  # surface (station) pressure (hPa)
                     "u",     # 10 m west-east wind (m/s)
                     "v",     # 10 m south-north wind (m/s)
                     "ws",    # 10 m scalar-mean wind speed (m/s)
                     "wd"]    # 10 m wind direction (degrees from)

MESONET_UNITS = {"t2": "K", "t9": "K", "q": "kg/kg", "td": "K", "psfc": "hPa", "u": "m/s", "v": "m/s", "ws": "m/s",
                 "wd": "degrees"}


# ---------------------------------------------------------------------------------------------------------------
# Time grids
# ---------------------------------------------------------------------------------------------------------------

def event_window(event: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end) of an event's window."""
    start = pd.to_datetime(event, format="%Y%m%d") + prepare_runs.INIT_OFFSET
    return start, start + pd.Timedelta(hours=24)


def tower_slots(event: str) -> pd.DatetimeIndex:
    """The 96 slots of an event's window, labeled by their start."""
    start, end = event_window(event)
    return pd.date_range(start, end - INTERVAL, freq=INTERVAL, name="slot_start")


def mesonet_times(event: str) -> pd.DatetimeIndex:
    """Every INTERVAL of an event's window, end included (97 times)."""
    start, end = event_window(event)
    return pd.date_range(start, end, freq=INTERVAL, name="time")


def centred_means(native: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Means of native's samples within INTERVAL / 2 of each of times ([T - INTERVAL / 2, T + INTERVAL / 2)), NaN with
    fewer than half the expected samples."""
    step = native.index.to_series().diff().median()
    needed = max(1, int(np.ceil(0.5 * (INTERVAL / step))))
    groups = native.groupby((native.index + INTERVAL / 2).floor(INTERVAL))
    return groups.mean().where(groups.count() >= needed).reindex(times)


# ---------------------------------------------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class TowerData:
    """TOWER_VARIABLES of one source (the observations or a run) at the booms of TOWER_HEIGHTS, one row per slot of
    the event window (tower_slots). A source lacks the variables it has no value for (see
    load_observations.TOWER_OBSERVED and transform_tower)."""
    source : str  # "obs", or the run's scheme
    event : str  # "YYYYMMDD"
    profiles : dict[str, pd.DataFrame] = field(repr=False)  # variable -> slot x boom
    sfc : pd.DataFrame = field(repr=False)  # slot x single-level variable (the model's surface layer; none observed)

    def __repr__(self):
        return (f"TowerData({self.source!r}, {self.event}, {len(self.times)} slots, {self.times[0]} -> "
                f"{self.times[-1]}, {len(self.variables)} variables)")

    def __getitem__(self, var: str) -> pd.DataFrame:
        return self.profiles[var]

    @property
    def variables(self) -> list[str]:
        return list(self.profiles)

    @property
    def times(self) -> pd.DatetimeIndex:
        return self.sfc.index

    def at(self, time) -> pd.DataFrame:
        """boom x [height (m AGL), *variables] at one slot."""
        time = pd.Timestamp(time)
        return pd.DataFrame({"height": pd.Series(TOWER_HEIGHTS)}
                            | {var: df.loc[time] for var, df in self.profiles.items()}).rename_axis("boom")


@dataclass
class MesonetPoints:
    """MESONET_VARIABLES of one source (the observations or a run) at the tslist's mesonet stations, at
    mesonet_times."""
    source : str  # "obs", or the run's scheme
    event : str  # "YYYYMMDD"
    stations : pd.DataFrame = field(repr=False)  # prefix -> name, latitude, longitude, elevation (m ASL), ...
    data : dict[str, pd.DataFrame] = field(repr=False)  # variable -> time x station

    def __repr__(self):
        return (f"MesonetPoints({self.source!r}, {self.event}, {len(self.stations)} stations, {len(self.times)} times, "
                f"{self.times[0]} -> {self.times[-1]})")

    def __getitem__(self, var: str) -> pd.DataFrame:
        return self.data[var]

    @property
    def variables(self) -> list[str]:
        return list(self.data)

    @property
    def times(self) -> pd.DatetimeIndex:
        return next(iter(self.data.values())).index

    def at(self, time) -> pd.DataFrame:
        """station x [latitude, longitude, *variables] at one time."""
        time = pd.Timestamp(time)
        values = pd.DataFrame({var: df.loc[time] for var, df in self.data.items()})
        return self.stations[["latitude", "longitude"]].join(values)

    def station(self, prefix: str) -> pd.DataFrame:
        """time x variable at one station."""
        return pd.DataFrame({var: df[prefix] for var, df in self.data.items()})

    def table(self) -> pd.DataFrame:
        """(time, station) x variable."""
        first = next(iter(self.data.values()))
        index = pd.MultiIndex.from_product([first.index, first.columns], names=["time", "station"])
        return pd.DataFrame({var: df.to_numpy().ravel() for var, df in self.data.items()}, index=index)
