"""
tower.py
Interpolate modeled tower-point profiles from model levels to the TTU tower's boom heights

Usage:
    import loader, tower
    run = loader.load_run("20131210", "mynn")
    df = tower.get_tower_df(run)                       # tower_column.nc (15 min) at booms 4-10 plus derived
                                                       # quantities: "u_4" ... "ustar_10", "heat_flux_sfc", "pblh", ...
    df = tower.interpolate_column(run.tower.column)   # only the interpolated tower_column.nc variables
    df = tower.interpolate_profiles(run.tower)         # .UU/.VV/.TH/... time series profiles (5 s)
    fluxes = tower.flux_profiles(run.tower.column)     # heat_flux, momt_flux on the model w levels

Output is wide, one column per variable and boom ("{var}_{boom}", the same naming as ttu_tower's wide table),
indexed by time. Interpolation is linear in height above ground, done separately at every time step, since
the model level heights move with time. Boom heights outside the range of the levels a variable lives on are
NaN (no extrapolation): mass-level variables start at ~4.6 m, so booms 1-3 (0.9, 2.4, 4.0 m) are NaN for them,
and the fluxes start at the second w level (~9 m). interpolate_column therefore only covers booms 4-10
(COLUMN_HEIGHTS); interpolate_profiles still returns every boom.

Turbulent fluxes (flux_profiles): local K-theory fluxes from the exchange coefficients and the mass-level
differences, at the interior w levels (k = 2 ... number of mass levels), where the PBL schemes keep K:
    heat_flux = -K_h (theta_k - theta_k-1) / (z_k - z_k-1)                       <w'theta'>, K m/s
    momt_flux = K_m sqrt((u_k - u_k-1)^2 + (v_k - v_k-1)^2) / (z_k - z_k-1)      |<u'w'>, <v'w'>|, m2/s2
These are only the local (down-gradient) part: the nonlocal terms of YSU (countergradient, entrainment), ACM2
(nonlocal upward mixing) and MYNN-EDMF (mass flux) are not included, so in a convective boundary layer they
underestimate the total flux. At the ground the fluxes come from the surface layer scheme instead:
    heat_flux_sfc = hfx / (rho c_p)    momt_flux_sfc = ust^2
with rho the air density at the lowest mass level. get_tower_df adds buoy_flux = heat_flux * vpt / theta, an
estimate of the buoyancy flux <w'theta_v'> that scales the heat flux by the mean moisture but leaves out the
moisture flux term, and ustar = sqrt(momt_flux), a local friction velocity (as in ttu_tower).

Winds: the earth-relative components u and v are interpolated, and ws and wd are computed from the
interpolated components. Interpolating ws or wd directly would be inconsistent with u, v (and wd wraps at 360).
"""

import numpy as np
import pandas as pd

import loader
from ttu_tower import constants
from ttu_tower.physics import thermo
from ttu_tower.physics.constants import R_CP, REFERENCE_PRESSURE, WATER_AIR_MWR

HEIGHTS: dict[int, float] = constants.HEIGHTS  # boom -> height, m AGL
COLUMN_HEIGHTS: dict[int, float] = {b: h for b, h in HEIGHTS.items() if b >= 4}  # booms within the column's levels

MASS_LEVEL_VARS = ["u", "v", "theta", "p", "qv"]  # TowerColumn variables on mass levels (heights column.z)
W_LEVEL_VARS = ["exch_h", "exch_m", "tke_pbl"]  # TowerColumn variables on w levels (heights column.z_w), but MYNN's
                                                # tke_pbl is on mass levels; interpolate_column checks each one's shape
RENAME = {"tke_pbl": "tke"}  # TowerColumn name -> output column prefix, where they differ
PROFILE_VARS = ["u", "v", "w", "theta", "p", "qv"]  # Station profile variables, all on mass levels (station.z)

# WRF's constants (share/module_model_constants.F)
R_D = 287.0  # J kg-1 K-1
CP = 7.0 * R_D / 2.0  # J kg-1 K-1
P0 = 1.0e5  # Pa, reference pressure of theta
XLV = 2.5e6  # J kg-1, latent heat of vaporization


def interpolate_to_heights(values: pd.DataFrame, z: pd.DataFrame, heights: dict[int, float] = HEIGHTS) -> pd.DataFrame:
    """Linearly interpolate a (time x level) profile with level heights z (time x level, m AGL, increasing with
    level) to fixed heights, separately at each time. Returns time x boom; NaN outside each time's level range."""
    if values.shape != z.shape:
        raise ValueError(f"values {values.shape} and heights {z.shape} differ in shape")
    targets = np.array(list(heights.values()), dtype=float)
    vals, zz = values.to_numpy(dtype=float), z.to_numpy(dtype=float)
    out = np.full((len(vals), len(targets)), np.nan)
    for i in range(len(vals)):
        out[i] = np.interp(targets, zz[i], vals[i], left=np.nan, right=np.nan)
    return pd.DataFrame(out, index=values.index, columns=pd.Index(list(heights), name="boom"))


def flux_profiles(column: loader.TowerColumn) -> dict[str, pd.DataFrame]:
    """Local K-theory fluxes at the interior w levels 2 ... n (n = number of mass levels), as time x level_w
    frames: "heat_flux" (K m/s), "momt_flux" (m2/s2), and "z" (their heights, m AGL, from column.z_w)."""
    z = column.z.to_numpy(dtype=float)
    dz = np.diff(z, axis=1)  # entry k - 2 (0-based) is between mass levels k - 1 and k, i.e. at w level k
    n = z.shape[1]
    dtheta = np.diff(column.theta.to_numpy(dtype=float), axis=1)
    du = np.diff(column.u.to_numpy(dtype=float), axis=1)
    dv = np.diff(column.v.to_numpy(dtype=float), axis=1)
    k_h = column.exch_h.to_numpy(dtype=float)[:, 1:n]  # w levels 2 ... n
    k_m = column.exch_m.to_numpy(dtype=float)[:, 1:n]
    levels = pd.RangeIndex(2, n + 1, name="level_w")

    def frame(data):
        return pd.DataFrame(data, index=column.z.index, columns=levels)

    return {"heat_flux": frame(-k_h * dtheta / dz),
            "momt_flux": frame(k_m * np.hypot(du, dv) / dz),
            "z": frame(column.z_w.to_numpy(dtype=float)[:, 1:n])}


def temperature(theta, p):
    """Temperature (K) from potential temperature (K) and pressure (Pa): the inverse of
    ttu_tower.physics.thermo.potential_temperature, with the same constants (p converted to kPa)."""
    return theta * (p / 1000.0 / REFERENCE_PRESSURE) ** R_CP


def surface_fluxes(column: loader.TowerColumn) -> pd.DataFrame:
    """Surface fluxes from the surface layer scheme, with rho, theta and qv at the lowest mass level:
        heat_flux_sfc       hfx / (rho c_p), <w'theta'> (K m/s)
        buoy_flux_sfc       heat_flux_sfc * vpt / theta, the same estimate as buoy_flux (no moisture flux term)
        real_buoy_flux_sfc  the full <w'theta_v'> (K m/s): buoy_flux_sfc + theta (1/eps - 1) / (1 + r)^2 <w'r'>, the
                            linearization of ttu_tower's exact vpt = theta (1 + r/eps) / (1 + r), with the vapor flux
                            <w'r'> = lh / (rho L_v)
        momt_flux_sfc       ust^2 (m2/s2)
    WRF's R_d, c_p and L_v, the constants WRF itself relates hfx and lh to the kinematic fluxes with."""
    p, theta, qv = column.p[1], column.theta[1], column.qv[1]
    t = theta * (p / P0) ** (R_D / CP)
    rho = p / (R_D * thermo.virtual_temperature(t, qv))
    heat_flux = column.sfc["hfx"] / (rho * CP)
    buoy_flux = heat_flux * thermo.virtual_potential_temperature(theta, qv) / theta
    vapor_flux = column.sfc["lh"] / (rho * XLV)  # kg/kg m/s
    real_buoy_flux = buoy_flux + theta * (1.0 / WATER_AIR_MWR - 1.0) / (1.0 + qv) ** 2 * vapor_flux
    return pd.DataFrame({"heat_flux_sfc": heat_flux, "buoy_flux_sfc": buoy_flux,
                         "real_buoy_flux_sfc": real_buoy_flux, "momt_flux_sfc": column.sfc["ust"] ** 2})


def _wide(interpolated: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """{var: time x boom} -> one wide frame with columns "{var}_{boom}", grouped by variable."""
    return pd.concat({var: df.set_axis([f"{var}_{b}" for b in df.columns], axis=1) for var, df in interpolated.items()},
                     axis=1).droplevel(0, axis=1)


def _add_wind(interpolated: dict[str, pd.DataFrame]) -> None:
    interpolated["ws"] = loader.wind_speed(interpolated["u"], interpolated["v"])
    interpolated["wd"] = loader.wind_direction(interpolated["u"], interpolated["v"])


def _interpolate_column(column: loader.TowerColumn, heights: dict[int, float]) -> dict[str, pd.DataFrame]:
    """{var: time x boom} for interpolate_column."""
    interpolated = {var: interpolate_to_heights(getattr(column, var), column.z, heights) for var in MASS_LEVEL_VARS}
    _add_wind(interpolated)
    for var in W_LEVEL_VARS:
        values = getattr(column, var)
        if values is not None:
            z = column.z_w if values.shape[1] == column.z_w.shape[1] else column.z  # w or mass levels
            interpolated[RENAME.get(var, var)] = interpolate_to_heights(values, z, heights)
    fluxes = flux_profiles(column)
    for var in ("heat_flux", "momt_flux"):
        interpolated[var] = interpolate_to_heights(fluxes[var], fluxes["z"], heights)
    return interpolated


def interpolate_column(column: loader.TowerColumn, heights: dict[int, float] = COLUMN_HEIGHTS) -> pd.DataFrame:
    """All height-dependent tower_column.nc variables at the boom heights (booms 4-10 by default; tke only for
    TKE schemes), plus ws and wd from the interpolated u and v, and heat_flux and momt_flux from flux_profiles."""
    return _wide(_interpolate_column(column, heights))


# column order of get_tower_df (variables a run does not have, e.g. tke for YSU, are skipped)
TOWER_DF_ORDER = ["u", "v", "ws", "wd", "theta", "t", "vpt", "p", "qv", "exch_h", "exch_m", "tke",
                  "heat_flux", "buoy_flux", "momt_flux", "ustar"]


def get_tower_df(run: loader.RunOutput, heights: dict[int, float] = COLUMN_HEIGHTS,
                 drop_initial: bool = True) -> pd.DataFrame:
    """The tower column of a run at the boom heights (interpolate_column), plus, per boom:
        t           temperature (K), from theta and p
        vpt         virtual potential temperature (K), from theta and qv (ttu_tower.physics.thermo, exact form)
        buoy_flux   estimated buoyancy flux <w'theta_v'> (K m/s): heat_flux * vpt / theta (moisture flux neglected)
        ustar       local friction velocity (m/s): sqrt(momt_flux)
    and the surface columns heat_flux_sfc, buoy_flux_sfc, real_buoy_flux_sfc, momt_flux_sfc (surface_fluxes),
    ustar_sfc = ust (m/s) and pblh (m).
    drop_initial removes the first time, the initial state written before the first model step: the PBL and
    surface layer schemes have not run yet, so e.g. TKE, the exchange coefficients and the fluxes are not set."""
    column = run.tower.column
    df = _interpolate_column(column, heights)
    df["t"] = temperature(df["theta"], df["p"])
    df["vpt"] = thermo.virtual_potential_temperature(df["theta"], df["qv"])
    df["buoy_flux"] = df["heat_flux"] * df["vpt"] / df["theta"]
    df["ustar"] = np.sqrt(df["momt_flux"])
    sfc = surface_fluxes(column)
    sfc["ustar_sfc"] = np.sqrt(sfc["momt_flux_sfc"])
    sfc["pblh"] = column.sfc["pblh"]
    out = pd.concat([_wide({var: df[var] for var in TOWER_DF_ORDER if var in df}), sfc], axis=1)
    return out.iloc[1:] if drop_initial else out


def interpolate_profiles(station: loader.Station, heights: dict[int, float] = HEIGHTS) -> pd.DataFrame:
    """The .UU/.VV/.WW/.TH/.PR/.QV profiles of a Station (e.g. run.tower) at the boom heights, plus ws and wd
    from the interpolated u and v. No exchange coefficients or TKE: they are not in the time series files."""
    interpolated = {var: interpolate_to_heights(getattr(station, var), station.z, heights) for var in PROFILE_VARS}
    _add_wind(interpolated)
    return _wide(interpolated)


if __name__ == "__main__":
    # hard-coded for now (development)
    DATE = "20131210"
    SCHEME = "mynn"

    run = loader.load_run(DATE, SCHEME)
    df = get_tower_df(run)
    print(df.filter(regex=r"^(ws|wd|theta|vpt|tke)_").iloc[:5].round(3).to_string())
    print(df.filter(regex=r"^(heat_flux|buoy_flux|ustar)_|^pblh$").iloc[:5].to_string())
