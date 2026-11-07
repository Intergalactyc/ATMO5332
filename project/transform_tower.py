"""
transform_tower.py
The modeled TTU tower of a run (its tslist time series and tower_column.nc) as slot values at the tower's boom
heights, comparable to the tower's slot averages

Usage:
    import load_results, transform_tower
    run = load_results.load_run("20131210", "mynn")
    df = transform_tower.get_tower_df(run)                 # 15-min slots at booms 4-10: "u_4" ... "ustar_10", "pblh", ...
    df = transform_tower.get_tower_df(run, slot="10min")   # 10-min slots, like the oneyear tower run

Output is wide, one column per variable and boom ("{var}_{boom}", e.g. "ws_4"; ttu_tower's wide table, as in
load_observations.load_tower_data, names them "{var}_b{boom}", e.g. "ws_mean_b4"), one row per slot [T, T + slot)
labeled T, the tower's slot_start convention. get_tower_df lists the columns and how each is made.

Heights: profiles are interpolated linearly in height above ground, separately at every time, since the model level
heights move with time, and not extrapolated. The lowest mass level is at ~4.6 m, so only booms 4-10 are covered
(booms 1-3 are at 0.9, 2.4 and 4.0 m), and the fluxes start at the second w level (~9 m).

Turbulent fluxes: local K-theory fluxes from the exchange coefficients and the mass-level differences, at the
interior w levels (k = 2 ... number of mass levels), where the PBL schemes keep K:
    heat_flux = -K_h (theta_k - theta_k-1) / (z_k - z_k-1)                       <w'theta'>, K m/s
    momt_flux = K_m sqrt((u_k - u_k-1)^2 + (v_k - v_k-1)^2) / (z_k - z_k-1)      |<u'w'>, <v'w'>|, m2/s2
These are only the local (down-gradient) part: the nonlocal terms of YSU (countergradient, entrainment), ACM2
(nonlocal upward mixing) and MYNN-EDMF (mass flux) are not included, so in a convective boundary layer they
underestimate the total flux. At the ground the fluxes come from the surface layer scheme instead:
    heat_flux_sfc = hfx / (rho c_p)    momt_flux_sfc = ust^2
with rho the air density at the lowest mass level. buoy_flux = heat_flux * vpt / theta estimates the buoyancy flux
<w'theta_v'> by scaling the heat flux with the mean moisture, leaving out the moisture flux term, and
ustar = sqrt(momt_flux) is a local friction velocity (as in ttu_tower).

Winds: the earth-relative components u and v are interpolated and averaged, and ws and wd are computed from the
mean components. Interpolating ws or wd directly would be inconsistent with u, v (and wd wraps at 360).
"""

import numpy as np
import pandas as pd

import load_results
from ttu_tower import constants
from ttu_tower.physics import thermo
from ttu_tower.physics.constants import R_CP, REFERENCE_PRESSURE, WATER_AIR_MWR

HEIGHTS: dict[int, float] = {b: h for b, h in constants.HEIGHTS.items() if b >= 4}  # boom -> m AGL, booms 4-10

AVERAGED_VARS = ["u", "v", "theta", "p", "qv"]  # tslist profiles averaged over each slot, on mass levels (station.z)
SNAPSHOT_VARS = ["exch_h", "exch_m", "tke_pbl"]  # TowerColumn profiles, on w levels (column.z_w), but MYNN's tke_pbl
                                                 # is on mass levels; get_tower_df checks each one's shape
RENAME = {"tke_pbl": "tke"}  # TowerColumn name -> output column prefix, where they differ

# column order of get_tower_df (variables a run does not have, e.g. tke for YSU, are skipped)
TOWER_DF_ORDER = ["u", "v", "ws", "ws_mean", "wd", "theta", "t", "vpt", "p", "qv", "exch_h", "exch_m", "tke",
                  "heat_flux", "buoy_flux", "momt_flux", "ustar"]

SLOT = pd.Timedelta("15min")  # default averaging slot (the d03 history interval)
MIN_COVERAGE = 0.75  # fraction of a slot's time series samples a slot mean needs (else NaN)

# WRF's constants (share/module_model_constants.F)
R_D = 287.0  # J kg-1 K-1
CP = 7.0 * R_D / 2.0  # J kg-1 K-1
P0 = 1.0e5  # Pa, reference pressure of theta
XLV = 2.5e6  # J kg-1, latent heat of vaporization


def _to_heights(values: pd.DataFrame, z: pd.DataFrame) -> pd.DataFrame:
    """time x level values with level heights z (time x level, m AGL, increasing) -> time x boom at HEIGHTS."""
    targets = np.array(list(HEIGHTS.values()))
    vals, zz = values.to_numpy(dtype=float), z.to_numpy(dtype=float)
    out = np.array([np.interp(targets, zz[i], vals[i], left=np.nan, right=np.nan) for i in range(len(vals))])
    return pd.DataFrame(out, index=values.index, columns=pd.Index(list(HEIGHTS), name="boom"))


def _to_times(df: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Rows of df (indexed by increasing time) linearly interpolated to times; NaN outside df's time range."""
    seconds = lambda t: (t - df.index[0]) / pd.Timedelta("1s")
    x, t = seconds(df.index), seconds(times)
    return pd.DataFrame({c: np.interp(t, x, df[c], left=np.nan, right=np.nan) for c in df}, index=times)


def _add_wind(frames: dict[str, pd.DataFrame]) -> None:
    frames["ws"] = load_results.wind_speed(frames["u"], frames["v"])
    frames["wd"] = load_results.wind_direction(frames["u"], frames["v"])


def _slot_means(station: load_results.Station, slot: pd.Timedelta) -> dict[str, pd.DataFrame]:
    """{var: slot x boom} means of AVERAGED_VARS and the scalar speed ws_mean over [T, T + slot), labeled T (each
    sample interpolated to the booms first), with ws, wd from the mean u, v. A mean is NaN with fewer than
    MIN_COVERAGE of the slot's samples; slots with no mean at all (the one-sample slot at the last output) are dropped."""
    frames = {var: _to_heights(getattr(station, var), station.z) for var in AVERAGED_VARS}
    frames["ws_mean"] = load_results.wind_speed(frames["u"], frames["v"])
    times = station.z.index
    needed = MIN_COVERAGE * (slot / times.to_series().diff().median())
    keys = times.floor(slot)
    means = {}
    for var, df in frames.items():
        groups = df.groupby(keys)
        means[var] = groups.mean().where(groups.count() >= needed)
    keep = pd.concat(means, axis=1).notna().any(axis=1)
    means = {var: df[keep] for var, df in means.items()}
    _add_wind(means)
    return means


def _flux_profiles(column: load_results.TowerColumn) -> dict[str, pd.DataFrame]:
    """Local K-theory fluxes (module docstring) at the interior w levels 2 ... n (n = number of mass levels), as
    time x level frames: "heat_flux" (K m/s), "momt_flux" (m2/s2), and "z" (their heights, m AGL, from column.z_w)."""
    z = column.z.to_numpy(dtype=float)
    dz = np.diff(z, axis=1)  # entry k - 2 (0-based) is between mass levels k - 1 and k, i.e. at w level k
    n = z.shape[1]
    dtheta = np.diff(column.theta.to_numpy(dtype=float), axis=1)
    du = np.diff(column.u.to_numpy(dtype=float), axis=1)
    dv = np.diff(column.v.to_numpy(dtype=float), axis=1)
    k_h = column.exch_h.to_numpy(dtype=float)[:, 1:n]  # w levels 2 ... n
    k_m = column.exch_m.to_numpy(dtype=float)[:, 1:n]

    def frame(data):
        return pd.DataFrame(data, index=column.z.index)

    return {"heat_flux": frame(-k_h * dtheta / dz),
            "momt_flux": frame(k_m * np.hypot(du, dv) / dz),
            "z": frame(column.z_w.to_numpy(dtype=float)[:, 1:n])}


def _surface(column: load_results.TowerColumn) -> pd.DataFrame:
    """Surface fluxes from the surface layer scheme, with rho, theta and qv at the lowest mass level, and pblh:
        heat_flux_sfc       hfx / (rho c_p), <w'theta'> (K m/s)
        buoy_flux_sfc       heat_flux_sfc * vpt / theta, the same estimate as buoy_flux (no moisture flux term)
        real_buoy_flux_sfc  the full <w'theta_v'> (K m/s): buoy_flux_sfc + theta (1/eps - 1) / (1 + r)^2 <w'r'>, the
                            linearization of ttu_tower's exact vpt = theta (1 + r/eps) / (1 + r), with the vapor flux
                            <w'r'> = lh / (rho L_v)
        momt_flux_sfc       ust^2 (m2/s2)
        pblh                PBL height (m)
    WRF's R_d, c_p and L_v, the constants WRF itself relates hfx and lh to the kinematic fluxes with."""
    p, theta, qv = column.p[1], column.theta[1], column.qv[1]
    t = theta * (p / P0) ** (R_D / CP)
    rho = p / (R_D * thermo.virtual_temperature(t, qv))
    heat_flux = column.sfc["hfx"] / (rho * CP)
    buoy_flux = heat_flux * thermo.virtual_potential_temperature(theta, qv) / theta
    vapor_flux = column.sfc["lh"] / (rho * XLV)  # kg/kg m/s
    real_buoy_flux = buoy_flux + theta * (1.0 / WATER_AIR_MWR - 1.0) / (1.0 + qv) ** 2 * vapor_flux
    return pd.DataFrame({"heat_flux_sfc": heat_flux, "buoy_flux_sfc": buoy_flux, "real_buoy_flux_sfc": real_buoy_flux,
                         "momt_flux_sfc": column.sfc["ust"] ** 2, "pblh": column.sfc["pblh"]})


def get_tower_df(run: load_results.RunOutput, slot=SLOT) -> pd.DataFrame:
    """The modeled tower at booms 4-10, one row per slot [T, T + slot) labeled T (15 min by default):
        u, v, theta, p, qv   means of the 5-s tslist profiles over the slot (each sample interpolated to the booms,
                             then averaged; NaN with less than MIN_COVERAGE of the slot's samples)
        ws, wd               vector-mean speed and direction, from the mean u and v (the tower's ws_vector_mean)
        ws_mean              scalar-mean speed (the tower's ws_mean)
        t                    temperature (K), from the mean theta and p
        vpt                  virtual potential temperature (K), from the mean theta and qv (ttu_tower.physics.thermo,
                             exact form)
    The variables only in tower_column.nc (instantaneous, every 15 min) are interpolated linearly in time to the
    slot centre T + slot / 2 from the snapshots around it (for 15-min slots, the mean of the snapshots at T and
    T + 15 min), and are NaN where the snapshots do not bracket the centre:
        exch_h, exch_m, tke, heat_flux, momt_flux   at the booms (tke only for TKE schemes)
        buoy_flux   heat_flux * vpt / theta (K m/s), with the slot-mean vpt and theta
        ustar       sqrt(momt_flux) (m/s)
        heat_flux_sfc, buoy_flux_sfc, real_buoy_flux_sfc, momt_flux_sfc, pblh   see _surface
        ustar_sfc   sqrt(momt_flux_sfc) (m/s)
    The first snapshot, the initial state written before the first model step, is left out: the PBL and surface
    layer schemes have not run yet, so e.g. TKE, the exchange coefficients and the fluxes are not set. The first
    slot therefore has its means but NaN for the snapshot variables."""
    slot = pd.Timedelta(slot)
    column = run.tower.column
    df = _slot_means(run.tower, slot)
    slots = df["u"].index
    centres = slots + slot / 2

    def at_centres(snapshots):  # without the initial state
        return _to_times(snapshots.iloc[1:], centres).set_axis(slots, axis=0)

    for var in SNAPSHOT_VARS:
        values = getattr(column, var)
        if values is not None:
            z = column.z_w if values.shape[1] == column.z_w.shape[1] else column.z  # w or mass levels
            df[RENAME.get(var, var)] = at_centres(_to_heights(values, z))
    fluxes = _flux_profiles(column)
    for var in ("heat_flux", "momt_flux"):
        df[var] = at_centres(_to_heights(fluxes[var], fluxes["z"]))
    sfc = at_centres(_surface(column))

    df["t"] = df["theta"] * (df["p"] / 1000.0 / REFERENCE_PRESSURE) ** R_CP  # inverse of thermo.potential_temperature
    df["vpt"] = thermo.virtual_potential_temperature(df["theta"], df["qv"])
    df["buoy_flux"] = df["heat_flux"] * df["vpt"] / df["theta"]
    df["ustar"] = np.sqrt(df["momt_flux"])
    sfc.insert(sfc.columns.get_loc("pblh"), "ustar_sfc", np.sqrt(sfc["momt_flux_sfc"]))
    wide = pd.concat({var: df[var].set_axis([f"{var}_{b}" for b in HEIGHTS], axis=1)
                      for var in TOWER_DF_ORDER if var in df}, axis=1).droplevel(0, axis=1)
    return pd.concat([wide, sfc], axis=1)


if __name__ == "__main__":
    run = load_results.load_run("20131210", "mynn")
    df = get_tower_df(run)
    print(df.filter(regex=r"^(ws|wd|theta|vpt|tke)_").iloc[:5].round(3).to_string())
    print(df.filter(regex=r"^(heat_flux|buoy_flux|ustar)_|^pblh$").iloc[:5].to_string())
