"""Download the ERA5 forcing (pressure and single levels) for each event's run period into era5/<event>/."""

import itertools
import os
from datetime import datetime, timedelta

import cdsapi

from prepare_runs import EVENTS, INIT_OFFSET, TEMPLATE, get_int

AREA = [48, -122, 18, -82]  # N, W, S, E
LEVELS = ["1","2","3","5","7","10","20","30","50","70","100","125","150","175","200",
          "225","250","300","350","400","450","500","550","600","650","700","750",
          "775","800","825","850","875","900","925","950","975","1000"]
PL_VARS = ["geopotential", "temperature", "u_component_of_wind",
           "v_component_of_wind", "relative_humidity"]
SL_VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature",
           "2m_dewpoint_temperature", "surface_pressure", "mean_sea_level_pressure",
           "skin_temperature", "sea_surface_temperature", "sea_ice_cover", "snow_depth",
           "land_sea_mask", "geopotential"]         + [f"soil_temperature_level_{i}" for i in range(1, 5)]         + [f"volumetric_soil_water_layer_{i}" for i in range(1, 5)]

run_hours = get_int(TEMPLATE.read_text().splitlines(), "time_control", "run_hours")
c = cdsapi.Client()
for ev in EVENTS:
    os.makedirs(f"era5/{ev}", exist_ok=True)
    start = datetime.strptime(ev, "%Y%m%d") + INIT_OFFSET
    times = [start + timedelta(hours=h) for h in range(run_hours + 1)]
    for day, hours in itertools.groupby(times, key=lambda t: t.date()):
        common = {"product_type": ["reanalysis"], "year": [f"{day:%Y}"], "month": [f"{day:%m}"], "day": [f"{day:%d}"],
                  "time": [f"{t:%H}:00" for t in hours], "area": AREA,
                  "data_format": "grib", "download_format": "unarchived"}
        c.retrieve("reanalysis-era5-pressure-levels",
                   {**common, "variable": PL_VARS, "pressure_level": LEVELS},
                   f"era5/{ev}/era5_pl_{day:%Y%m%d}.grib")
        c.retrieve("reanalysis-era5-single-levels",
                   {**common, "variable": SL_VARS},
                   f"era5/{ev}/era5_sl_{day:%Y%m%d}.grib")
