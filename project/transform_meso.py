"""
transform_meso.py
Mesonet observations (load_observations) and a run's modeled mesonet stations (load_results), each as the same
variables at the same 15-min times and stations: one time x station frame per variable.

Usage:
    import load_observations, load_results, transform_meso
    obs = transform_meso.get_mesonet_observations(load_observations.load_mesonet_data("20131210"))
    model = transform_meso.get_mesonet_model(load_results.load_run("20131210", "ysu"))
    obs["t2"]                                   # time x station (tslist prefix)
    snap = model.at("2013-12-10 06:00")         # station x [latitude, longitude, *variables], for a map
    ax.scatter(snap.longitude, snap.latitude, c=snap.t2, transform=ccrs.PlateCarree())
    diff = transform_meso.difference(model, obs)    # model - obs, wd wrapped to [-180, 180)
    diff["t2"].abs().mean()                     # e.g. MAE per station
    obs.table()                                 # (time, station) x variable, for groupby-style statistics

Both sources hold every station of the tslist (prepare_runs.select_stations) in the same order, and every INTERVAL of
the event window (structs.mesonet_times, end included: 97 times), so frames of the two align exactly.
Stations without observations are all NaN in obs. latitude, longitude and elevation (m ASL) in .stations are the
station's own (meta/wtm_2013_station_locations.csv) for both; model.stations also has the grid point's.

The value at T is the mean of the samples within 7.5 min of T ([T - 7.5 min, T + 7.5 min)), NaN with fewer than half
the expected samples (structs.centred_means): for the observations as in load_observations, for the model of its 5-s
.TS (and tslist profile) samples. u and v are vector means, ws is the scalar-mean speed, and wd is from the mean u and
v. td is computed from the means of q and psfc. The model's first time (its initialization) is NaN, since only half its
window is simulated.

Model values come from the .TS surface series, except t9, which is interpolated linearly in height between the tslist
profile levels (the lowest is ~4.6 m AGL). The model's psfc is reduced hypsometrically from the grid point's terrain
height to the station's elevation (they differ by up to ~60 m, i.e. ~7 hPa) using its 2 m virtual temperature, unless
get_mesonet_model(adjust_pressure=False). Temperatures are not adjusted for elevation.
"""

import numpy as np
import pandas as pd
from ttu_tower.math.polar import vector_to_bearing
from ttu_tower.physics import thermo
from wtxmeso import qc

import load_observations
import load_results
import prepare_runs
from load_results import R_D, temperature
from structs import MESONET_VARIABLES, MesonetPoints, centred_means, mesonet_times

T9_HEIGHT = 9.0  # m AGL


def station_table() -> pd.DataFrame:
    """The tslist's mesonet stations (prefix) -> name, latitude, longitude, elevation (m ASL)."""
    selected = {name: pfx for name, pfx, _, _ in prepare_runs.select_stations()}
    meta = pd.read_csv(prepare_runs.STATIONS).set_index("archive_name").loc[list(selected)]
    return pd.DataFrame({"name": meta["Location"].to_numpy(),
                         "latitude": meta["latitude"].to_numpy(),
                         "longitude": meta["longitude"].to_numpy(),
                         "elevation": meta["elevation_ft"].to_numpy() * 0.3048},
                        index=pd.Index(list(selected.values()), name="station"))


def _dewpoint(t, q, psfc):
    """Dewpoint (K) from mixing ratio q (kg/kg) and pressure psfc (hPa); t (K) only passes through thermo's
    rh-based form, which inverts the same Magnus fit, so td does not depend on it."""
    e = q * psfc / (thermo.WATER_AIR_MWR + q) / 10.0  # kPa
    return thermo.dewpoint_temperature(t, e / thermo.saturation_vapor_pressure(t))


def _wind_direction(u, v):
    return vector_to_bearing(u, v)[1]


def _at_height(values: pd.DataFrame, z: pd.DataFrame, height: float) -> pd.Series:
    """time x level values at heights z (time x level, m AGL, increasing), interpolated linearly to height."""
    vals, zz = values.to_numpy(dtype=float), z.to_numpy(dtype=float)
    below = (zz <= height).sum(axis=1) - 1
    inside = (below >= 0) & (below < zz.shape[1] - 1)
    k = np.clip(below, 0, zz.shape[1] - 2)
    rows = np.arange(len(zz))
    z0, z1, v0, v1 = zz[rows, k], zz[rows, k + 1], vals[rows, k], vals[rows, k + 1]
    out = np.where(inside, v0 + (v1 - v0) * (height - z0) / (z1 - z0), np.nan)
    return pd.Series(out, index=values.index)


def _assemble(source: str, event: str, stations: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> MesonetPoints:
    """{station: time x variable} -> MesonetPoints over all of stations."""
    times = mesonet_times(event)
    empty = pd.DataFrame(np.nan, index=times, columns=MESONET_VARIABLES)
    data = {var: pd.DataFrame({pfx: frames.get(pfx, empty)[var] for pfx in stations.index}, index=times)
            .rename_axis(columns="station") for var in MESONET_VARIABLES}
    return MesonetPoints(source=source, event=event, stations=stations, data=data)


def get_mesonet_observations(data: load_observations.MesonetObservations) -> MesonetPoints:
    """The observed MESONET_VARIABLES: load_observations' 15-min values, plus the QC'd 9 m temperature averaged the
    same way and td."""
    times = mesonet_times(data.event)
    frames = {}
    for pfx, st in data.stations.items():
        df = st.df.reindex(times)
        clean = qc.apply(st.raw, st.flags)
        clean[st.excluded] = np.nan
        df["t9"] = centred_means(clean[["9m Temp"]] + 273.15, times)["9m Temp"]
        df["td"] = _dewpoint(df["t"], df["q"], df["psfc"])
        frames[pfx] = df.rename(columns={"t": "t2"})[MESONET_VARIABLES]
    return _assemble("obs", data.event, station_table(), frames)


def get_mesonet_model(run: load_results.RunOutput, adjust_pressure: bool = True) -> MesonetPoints:
    """The modeled MESONET_VARIABLES at the mesonet stations: 15-min means of the .TS series and of t9 from the tslist
    profiles, with psfc at the station elevation if adjust_pressure."""
    stations = station_table()
    times = mesonet_times(run.event)
    frames, grid = {}, {}
    for pfx in stations.index:
        st = run.stations.get(pfx)
        if st is None:
            continue
        grid[pfx] = (st.latitude, st.longitude, st.elevation)
        native = st.df[["t", "q", "psfc", "u", "v", "ws"]].copy()
        native["t9"] = _at_height(temperature(st.theta, st.p), st.z, T9_HEIGHT).reindex(native.index)
        df = centred_means(native, times)
        df["wd"] = _wind_direction(df["u"], df["v"])
        df["td"] = _dewpoint(df["t"], df["q"], df["psfc"])
        if adjust_pressure:
            dz = stations.at[pfx, "elevation"] - st.elevation
            df["psfc"] *= np.exp(-thermo.STANDARD_GRAVITY * dz / (R_D * thermo.virtual_temperature(df["t"], df["q"])))
        frames[pfx] = df.rename(columns={"t": "t2"})[MESONET_VARIABLES]
    grid = pd.DataFrame.from_dict(grid, orient="index", columns=["grid_latitude", "grid_longitude", "grid_elevation"])
    return _assemble(run.scheme, run.event, stations.join(grid), frames)


def difference(a: MesonetPoints, b: MesonetPoints) -> MesonetPoints:
    """a - b (e.g. model - obs) for each variable both have, with wd wrapped to [-180, 180)."""
    if a.event != b.event:
        raise ValueError(f"different events: {a.event}, {b.event}")
    data = {var: a[var] - b[var] for var in a.variables if var in b.data}
    if "wd" in data:
        data["wd"] = (data["wd"] + 180.0) % 360.0 - 180.0
    return MesonetPoints(source=f"{a.source} - {b.source}", event=a.event, stations=a.stations, data=data)


if __name__ == "__main__":
    obs = get_mesonet_observations(load_observations.load_mesonet_data("20131210"))
    model = get_mesonet_model(load_results.load_run("20131210", "ysu"))
    print(obs, model, sep="\n")
    print(model.at("2013-12-10 06:00").head().round(3).to_string())
    diff = difference(model, obs)
    print(pd.DataFrame({"bias": diff.table().mean(), "mae": diff.table().abs().mean(),
                        "rmse": np.sqrt((diff.table() ** 2).mean())}).round(3).to_string())
