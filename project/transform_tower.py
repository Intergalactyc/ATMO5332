"""
transform_tower.py
A run's modeled tower (tslist profiles and tower_column.nc) as 15-min slot values at the tower's booms, in the
observed tower's structure (structs.TowerData).

Usage:
    import load_observations, load_results, transform_tower
    model = transform_tower.get_tower_model(load_results.load_run("20131210", "mynn"))
    obs = load_observations.load_tower_data("20131210")
    model["ws"] - obs["ws"]                     # slot x boom
    model.sfc["pblh"]                           # surface layer scheme values and pblh

Slots are [T, T + 15 min), labeled T, as the observations'. Profiles are interpolated linearly in height at each time,
without extrapolation: booms 1-3 (0.9-4 m) are below the lowest mass level (~4.6 m) and stay NaN, and the fluxes
start at the second w level (~9 m), so boom 4 (10.1 m) has them. Winds are averaged as u and v, with ws and wd from the
means. Pressures are in hPa.

Stability, as ttu-tower-processing computes it for the observations:
    obukhov_length = -ustar^3 vpt / (kappa g buoy_flux)    ttu_tower.physics.most, kappa = 0.41, g local at the boom
    rig = N^2 / shear^2, N^2 = g / vpt dvpt/dz              ttu_tower.tertiary.gradients, RIG_STENCIL's log-height
                                                            gradients over the booms the model reaches (4-10)
obukhov_length_sfc in sfc is the same from the surface layer scheme's ust and real_buoy_flux_sfc.

Fluxes are local K-theory fluxes at the interior w levels:
    heat_flux = -K_h dtheta/dz          <w'theta'>, K m/s
    momt_flux = K_m |dU/dz|             |<u'w'>, <v'w'>|, m2/s2
They leave out the nonlocal terms of YSU, ACM2 and MYNN-EDMF, so they underestimate convective boundary layer fluxes.
"""

import numpy as np
import pandas as pd
from ttu_tower.constants import SITE_ELEVATION, SITE_LATITUDE
from ttu_tower.math.polar import vector_to_bearing
from ttu_tower.physics import most, thermo
from ttu_tower.physics.constants import WATER_AIR_MWR
from ttu_tower.physics.earth import local_gravity
from ttu_tower.tertiary import gradients

import load_results
from load_results import CP, R_D, XLV, temperature
from structs import INTERVAL, TOWER_HEIGHTS, TOWER_VARIABLES, TowerData, tower_slots

AVERAGED_VARS = ["u", "v", "theta", "p", "qv"]  # tslist profiles, on mass levels
SNAPSHOT_VARS = ["exch_h", "exch_m", "tke_pbl"]  # TowerColumn profiles, on w levels (MYNN's tke_pbl on mass levels)
RENAME = {"tke_pbl": "tke"}

MIN_COVERAGE = 0.75  # fraction of a slot's samples a slot mean needs

RIG_STENCIL = "log_linear"  # ttu_tower's first (default) stencil, which its rig_b{b} columns use
GRAVITY = pd.Series({b: local_gravity(SITE_LATITUDE, SITE_ELEVATION + h) for b, h in TOWER_HEIGHTS.items()})  # m s-2


def _finite(x):
    """x with infinities (zero denominators) as NaN, as ttu_tower gives them."""
    return x.where(np.isfinite(x))


def _to_heights(values: pd.DataFrame, z: pd.DataFrame) -> pd.DataFrame:
    """time x level values at heights z (time x level, m AGL, increasing) -> time x boom."""
    targets = np.array(list(TOWER_HEIGHTS.values()))
    vals, zz = values.to_numpy(dtype=float), z.to_numpy(dtype=float)
    out = np.array([np.interp(targets, zz[i], vals[i], left=np.nan, right=np.nan) for i in range(len(vals))])
    return pd.DataFrame(out, index=values.index, columns=pd.Index(list(TOWER_HEIGHTS), name="boom"))


def _at_centres(snapshots: pd.DataFrame, slots: pd.DatetimeIndex) -> pd.DataFrame:
    """15-min snapshots at the slot centres T + 7.5 min: the mean of those at T and T + 15 min. The initial snapshot
    is left out, since the PBL and surface layer schemes have not run yet."""
    s = snapshots.iloc[1:]
    return ((s + s.shift(-1)) / 2).reindex(slots)


def _add_wind(frames: dict[str, pd.DataFrame]) -> None:
    frames["ws"], frames["wd"] = vector_to_bearing(frames["u"], frames["v"])


def _slot_means(station: load_results.Station, slots: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """{var: slot x boom} means of AVERAGED_VARS and ws_mean, with ws and wd from the mean u and v."""
    frames = {var: _to_heights(getattr(station, var), station.z) for var in AVERAGED_VARS}
    frames["ws_mean"] = np.hypot(frames["u"], frames["v"])
    times = station.z.index
    needed = MIN_COVERAGE * (INTERVAL / times.to_series().diff().median())
    keys = times.floor(INTERVAL)
    means = {}
    for var, df in frames.items():
        groups = df.groupby(keys)
        means[var] = groups.mean().where(groups.count() >= needed).reindex(slots)
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


def _richardson(u: pd.DataFrame, v: pd.DataFrame, vpt: pd.DataFrame) -> pd.DataFrame:
    """Gradient Richardson number (slot x boom), as ttu_tower.tertiary.gradients' rig, over the booms with values."""
    booms = list(vpt.columns[vpt.notna().any()])
    weights = gradients.stencil(booms, RIG_STENCIL)

    def ddz(x):
        return pd.DataFrame({b: x[points].to_numpy() @ w for b, (points, w) in weights.items()}, index=x.index)

    n2 = GRAVITY / vpt * ddz(vpt)
    return _finite(n2 / (ddz(u) ** 2 + ddz(v) ** 2)).reindex(columns=vpt.columns)


def _surface(column: load_results.TowerColumn) -> pd.DataFrame:
    """Surface layer scheme values, with rho, theta and qv (r) at the lowest mass level:
        heat_flux_sfc       hfx / (rho c_p), K m/s
        buoy_flux_sfc       heat_flux_sfc * vpt / theta, K m/s
        real_buoy_flux_sfc  buoy_flux_sfc + theta (1/eps - 1) / (1 + r)^2 <w'r'>, with <w'r'> = lh / (rho L_v): the
                            full <w'theta_v'> (K m/s), linearized from ttu_tower's vpt
        momt_flux_sfc       ust^2, m2/s2
        obukhov_length_sfc  -ust^3 vpt / (kappa g real_buoy_flux_sfc), m, g at the ground
        pblh                m"""
    p, theta, qv = column.p[1], column.theta[1], column.qv[1]
    rho = 100.0 * p / (R_D * thermo.virtual_temperature(temperature(theta, p), qv))  # kg m-3
    vpt = thermo.virtual_potential_temperature(theta, qv)
    heat_flux = column.sfc["hfx"] / (rho * CP)
    buoy_flux = heat_flux * vpt / theta
    vapor_flux = column.sfc["lh"] / (rho * XLV)
    real_buoy_flux = buoy_flux + theta * (1.0 / WATER_AIR_MWR - 1.0) / (1.0 + qv) ** 2 * vapor_flux
    obukhov_length = most.obukhov_length(column.sfc["ust"], vpt, real_buoy_flux,
                                         gravity=local_gravity(SITE_LATITUDE, SITE_ELEVATION))
    return pd.DataFrame({"heat_flux_sfc": heat_flux, "buoy_flux_sfc": buoy_flux, "real_buoy_flux_sfc": real_buoy_flux,
                         "momt_flux_sfc": column.sfc["ust"] ** 2, "obukhov_length_sfc": _finite(obukhov_length),
                         "pblh": column.sfc["pblh"]})


def get_tower_model(run: load_results.RunOutput) -> TowerData:
    """The modeled tower, one row per slot (booms 1-3 NaN):
        u, v, theta, p, qv      means of the 5-s tslist profiles (NaN with less than MIN_COVERAGE of the samples)
        ws, wd                  from the mean u and v (as the tower's ws_vector_mean)
        ws_mean                 scalar-mean speed
        t, vpt                  from the mean theta, p and qv
        exch_h, exch_m, tke     tower_column.nc at the slot centre (tke for TKE schemes only)
        heat_flux, momt_flux    the same
        buoy_flux               heat_flux * vpt / theta, leaving out the moisture flux
        ustar                   sqrt(momt_flux)
        obukhov_length, rig     from those (see the module docstring)
    and in sfc, at the slot centre: _surface's columns and ustar_sfc = sqrt(momt_flux_sfc). The first slot has no
    tower_column.nc values (see _at_centres)."""
    column = run.tower.column
    slots = tower_slots(run.event)
    df = _slot_means(run.tower, slots)

    for var in SNAPSHOT_VARS:
        values = getattr(column, var)
        if values is not None:
            z = column.z_w if values.shape[1] == column.z_w.shape[1] else column.z
            df[RENAME.get(var, var)] = _at_centres(_to_heights(values, z), slots)
    fluxes = _flux_profiles(column)
    for var in ("heat_flux", "momt_flux"):
        df[var] = _at_centres(_to_heights(fluxes[var], fluxes["z"]), slots)
    sfc = _at_centres(_surface(column), slots)

    df["t"] = temperature(df["theta"], df["p"])
    df["vpt"] = thermo.virtual_potential_temperature(df["theta"], df["qv"])
    df["buoy_flux"] = df["heat_flux"] * df["vpt"] / df["theta"]
    df["ustar"] = np.sqrt(df["momt_flux"])
    df["obukhov_length"] = _finite(most.obukhov_length(df["ustar"], df["vpt"], df["buoy_flux"], gravity=GRAVITY))
    df["rig"] = _richardson(df["u"], df["v"], df["vpt"])
    sfc.insert(sfc.columns.get_loc("pblh"), "ustar_sfc", np.sqrt(sfc["momt_flux_sfc"]))
    profiles = {var: df[var] for var in TOWER_VARIABLES if var in df}
    return TowerData(source=run.scheme, event=run.event, profiles=profiles, sfc=sfc)


if __name__ == "__main__":
    model = get_tower_model(load_results.load_run("20131210", "mynn"))
    print(model)
    print(model.at("2013-12-10 06:00").round(4).to_string())
    print(model.sfc.iloc[:5].to_string())
