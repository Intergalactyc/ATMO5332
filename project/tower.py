# tower.py
# Interpolate modeled tower-point profiles from model levels to the TTU tower's boom heights
#
# Usage:
#     import loader, tower
#     run = loader.load_run("20131210", "mynn")
#     df = tower.interpolate_column(run.tower.column)   # tower_column.nc (15 min): "u_1" ... "tke_pbl_10", "pblh", ...
#     df = tower.interpolate_profiles(run.tower)         # .UU/.VV/.TH/... time series profiles (5 s)
#
# Output is wide, one column per variable and boom ("{var}_{boom}", the same naming as ttu_tower's wide table),
# indexed by time. Interpolation is linear in height above ground, done separately at every time step, since
# the model level heights move with time. Boom heights outside the range of the levels a variable lives on are
# NaN (no extrapolation): mass-level variables start at ~4.6 m, so booms 1-3 (0.9, 2.4, 4.0 m) are NaN for them,
# while w-level variables (exch_h, exch_m, tke_pbl) start at the ground and cover every boom.
#
# Winds: the earth-relative components u and v are interpolated, and ws and wd are computed from the
# interpolated components. Interpolating ws or wd directly would be inconsistent with u, v (and wd wraps at 360).

import numpy as np
import pandas as pd

import loader
from ttu_tower import constants

HEIGHTS: dict[int, float] = constants.HEIGHTS  # boom -> height, m AGL

MASS_LEVEL_VARS = ["u", "v", "theta", "p", "qv"]  # TowerColumn variables on mass levels (heights column.z)
W_LEVEL_VARS = ["exch_h", "exch_m", "tke_pbl"]  # TowerColumn variables on w levels (heights column.z_w)
PROFILE_VARS = ["u", "v", "w", "theta", "p", "qv"]  # Station profile variables, all on mass levels (station.z)


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


def _wide(interpolated: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """{var: time x boom} -> one wide frame with columns "{var}_{boom}", grouped by variable."""
    return pd.concat({var: df.set_axis([f"{var}_{b}" for b in df.columns], axis=1) for var, df in interpolated.items()},
                     axis=1).droplevel(0, axis=1)


def _add_wind(interpolated: dict[str, pd.DataFrame]) -> None:
    interpolated["ws"] = loader.wind_speed(interpolated["u"], interpolated["v"])
    interpolated["wd"] = loader.wind_direction(interpolated["u"], interpolated["v"])


def interpolate_column(column: loader.TowerColumn, heights: dict[int, float] = HEIGHTS) -> pd.DataFrame:
    """All height-dependent tower_column.nc variables at the boom heights (tke_pbl only for TKE schemes), plus
    ws and wd from the interpolated u and v, plus the surface variables (pblh, ust, hfx, lh) as-is."""
    interpolated = {var: interpolate_to_heights(getattr(column, var), column.z, heights) for var in MASS_LEVEL_VARS}
    _add_wind(interpolated)
    for var in W_LEVEL_VARS:
        values = getattr(column, var)
        if values is not None:
            interpolated[var] = interpolate_to_heights(values, column.z_w, heights)
    return pd.concat([_wide(interpolated), column.sfc], axis=1)


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
    df = interpolate_column(run.tower.column)
    print(df.filter(regex=r"^(ws|wd|theta|tke_pbl)_").iloc[:5].round(3).to_string())
