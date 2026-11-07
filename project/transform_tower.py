"""
transform_tower.py
A run's modeled tower (tslist profiles and tower_column.nc) as 15-min slot values at the tower's boom heights,
comparable to the observed tower (load_observations).

Usage:
    import load_results, transform_tower
    df = transform_tower.get_tower_df(load_results.load_run("20131210", "mynn"))

Columns are "{var}_{boom}", e.g. "ws_4" (the observed tower's are "{var}_b{boom}"); rows are slots [T, T + 15 min)
labeled T. Profiles are interpolated linearly in height at each time, without extrapolation: the lowest mass level
(~4.6 m) leaves booms 4-10, and the fluxes start at the second w level (~9 m). Winds are averaged as u and v, with
ws and wd from the means.

Fluxes are local K-theory fluxes at the interior w levels:
    heat_flux = -K_h dtheta/dz          <w'theta'>, K m/s
    momt_flux = K_m |dU/dz|             |<u'w'>, <v'w'>|, m2/s2
They leave out the nonlocal terms of YSU, ACM2 and MYNN-EDMF, so they underestimate convective boundary layer fluxes.
"""

import numpy as np
import pandas as pd
from ttu_tower import constants
from ttu_tower.math.polar import vector_to_bearing
from ttu_tower.physics import thermo
from ttu_tower.physics.constants import WATER_AIR_MWR

import load_results
from load_results import INTERVAL

HEIGHTS: dict[int, float] = {b: h for b, h in constants.HEIGHTS.items() if b >= 4}  # boom -> m AGL

AVERAGED_VARS = ["u", "v", "theta", "p", "qv"]  # tslist profiles, on mass levels
SNAPSHOT_VARS = ["exch_h", "exch_m", "tke_pbl"]  # TowerColumn profiles, on w levels (MYNN's tke_pbl on mass levels)
RENAME = {"tke_pbl": "tke"}

# get_tower_df column order (variables a run lacks, e.g. YSU's tke, are skipped)
TOWER_DF_ORDER = ["u", "v", "ws", "ws_mean", "wd", "theta", "t", "vpt", "p", "qv", "exch_h", "exch_m", "tke",
                  "heat_flux", "buoy_flux", "momt_flux", "ustar"]

MIN_COVERAGE = 0.75  # fraction of a slot's samples a slot mean needs

# WRF's constants (share/module_model_constants.F)
R_D = 287.0  # J kg-1 K-1
CP = 7.0 * R_D / 2.0  # J kg-1 K-1
P0 = 1.0e5  # Pa
XLV = 2.5e6  # J kg-1


def _temperature(theta, p):
    """Temperature (K) from theta (K) and p (Pa)."""
    return theta * (p / P0) ** (R_D / CP)


def _to_heights(values: pd.DataFrame, z: pd.DataFrame) -> pd.DataFrame:
    """time x level values at heights z (time x level, m AGL, increasing) -> time x boom."""
    targets = np.array(list(HEIGHTS.values()))
    vals, zz = values.to_numpy(dtype=float), z.to_numpy(dtype=float)
    out = np.array([np.interp(targets, zz[i], vals[i], left=np.nan, right=np.nan) for i in range(len(vals))])
    return pd.DataFrame(out, index=values.index, columns=pd.Index(list(HEIGHTS), name="boom"))


def _at_centres(snapshots: pd.DataFrame, slots: pd.DatetimeIndex) -> pd.DataFrame:
    """15-min snapshots at the slot centres T + 7.5 min: the mean of those at T and T + 15 min. The initial snapshot
    is left out, since the PBL and surface layer schemes have not run yet."""
    s = snapshots.iloc[1:]
    return ((s + s.shift(-1)) / 2).reindex(slots)


def _add_wind(frames: dict[str, pd.DataFrame]) -> None:
    frames["ws"], frames["wd"] = vector_to_bearing(frames["u"], frames["v"])


def _slot_means(station: load_results.Station) -> dict[str, pd.DataFrame]:
    """{var: slot x boom} means of AVERAGED_VARS and ws_mean, with ws and wd from the mean u and v. Slots without
    any mean (the one-sample slot at the last output) are dropped."""
    frames = {var: _to_heights(getattr(station, var), station.z) for var in AVERAGED_VARS}
    frames["ws_mean"] = np.hypot(frames["u"], frames["v"])
    times = station.z.index
    needed = MIN_COVERAGE * (INTERVAL / times.to_series().diff().median())
    keys = times.floor(INTERVAL)
    means = {}
    for var, df in frames.items():
        groups = df.groupby(keys)
        means[var] = groups.mean().where(groups.count() >= needed)
    keep = pd.concat(means, axis=1).notna().any(axis=1)
    means = {var: df[keep] for var, df in means.items()}
    _add_wind(means)
    return means


def _flux_profiles(column: load_results.TowerColumn) -> dict[str, pd.DataFrame]:
    """heat_flux and momt_flux (time x level) at w levels 2 ... n (n mass levels), and their heights z."""
    z = column.z.to_numpy(dtype=float)
    dz = np.diff(z, axis=1)  # column k - 2 (0-based) is at w level k
    n = z.shape[1]
    dtheta = np.diff(column.theta.to_numpy(dtype=float), axis=1)
    du = np.diff(column.u.to_numpy(dtype=float), axis=1)
    dv = np.diff(column.v.to_numpy(dtype=float), axis=1)
    k_h = column.exch_h.to_numpy(dtype=float)[:, 1:n]
    k_m = column.exch_m.to_numpy(dtype=float)[:, 1:n]

    def frame(data):
        return pd.DataFrame(data, index=column.z.index)

    return {"heat_flux": frame(-k_h * dtheta / dz),
            "momt_flux": frame(k_m * np.hypot(du, dv) / dz),
            "z": frame(column.z_w.to_numpy(dtype=float)[:, 1:n])}


def _surface(column: load_results.TowerColumn) -> pd.DataFrame:
    """Surface layer scheme values, with rho, theta and qv (r) at the lowest mass level:
        heat_flux_sfc       hfx / (rho c_p), K m/s
        buoy_flux_sfc       heat_flux_sfc * vpt / theta, K m/s
        real_buoy_flux_sfc  buoy_flux_sfc + theta (1/eps - 1) / (1 + r)^2 <w'r'>, with <w'r'> = lh / (rho L_v): the
                            full <w'theta_v'> (K m/s), linearized from ttu_tower's vpt
        momt_flux_sfc       ust^2, m2/s2
        pblh                m"""
    p, theta, qv = column.p[1], column.theta[1], column.qv[1]
    rho = p / (R_D * thermo.virtual_temperature(_temperature(theta, p), qv))
    heat_flux = column.sfc["hfx"] / (rho * CP)
    buoy_flux = heat_flux * thermo.virtual_potential_temperature(theta, qv) / theta
    vapor_flux = column.sfc["lh"] / (rho * XLV)
    real_buoy_flux = buoy_flux + theta * (1.0 / WATER_AIR_MWR - 1.0) / (1.0 + qv) ** 2 * vapor_flux
    return pd.DataFrame({"heat_flux_sfc": heat_flux, "buoy_flux_sfc": buoy_flux, "real_buoy_flux_sfc": real_buoy_flux,
                         "momt_flux_sfc": column.sfc["ust"] ** 2, "pblh": column.sfc["pblh"]})


def get_tower_df(run: load_results.RunOutput) -> pd.DataFrame:
    """The modeled tower at booms 4-10, one row per slot:
        u, v, theta, p, qv      means of the 5-s tslist profiles (NaN with less than MIN_COVERAGE of the samples)
        ws, wd                  from the mean u and v (as the tower's ws_vector_mean)
        ws_mean                 scalar-mean speed
        t, vpt                  from the mean theta, p and qv
        exch_h, exch_m, tke     tower_column.nc at the slot centre (tke for TKE schemes only)
        heat_flux, momt_flux    the same
        buoy_flux               heat_flux * vpt / theta, leaving out the moisture flux
        ustar                   sqrt(momt_flux)
    and at the surface, at the slot centre: _surface's columns and ustar_sfc = sqrt(momt_flux_sfc). The first slot
    has no tower_column.nc values (see _at_centres)."""
    column = run.tower.column
    df = _slot_means(run.tower)
    slots = df["u"].index

    for var in SNAPSHOT_VARS:
        values = getattr(column, var)
        if values is not None:
            z = column.z_w if values.shape[1] == column.z_w.shape[1] else column.z
            df[RENAME.get(var, var)] = _at_centres(_to_heights(values, z), slots)
    fluxes = _flux_profiles(column)
    for var in ("heat_flux", "momt_flux"):
        df[var] = _at_centres(_to_heights(fluxes[var], fluxes["z"]), slots)
    sfc = _at_centres(_surface(column), slots)

    df["t"] = _temperature(df["theta"], df["p"])
    df["vpt"] = thermo.virtual_potential_temperature(df["theta"], df["qv"])
    df["buoy_flux"] = df["heat_flux"] * df["vpt"] / df["theta"]
    df["ustar"] = np.sqrt(df["momt_flux"])
    sfc.insert(sfc.columns.get_loc("pblh"), "ustar_sfc", np.sqrt(sfc["momt_flux_sfc"]))
    wide = pd.concat({var: df[var].set_axis([f"{var}_{b}" for b in HEIGHTS], axis=1)
                      for var in TOWER_DF_ORDER if var in df}, axis=1).droplevel(0, axis=1)
    return pd.concat([wide, sfc], axis=1)


if __name__ == "__main__":
    df = get_tower_df(load_results.load_run("20131210", "mynn"))
    print(df.filter(regex=r"^(ws|wd|theta|vpt|tke)_").iloc[:5].round(3).to_string())
    print(df.filter(regex=r"^(heat_flux|buoy_flux|ustar)_|^pblh$").iloc[:5].to_string())
