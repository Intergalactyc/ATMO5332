import cdsapi, os

EVENTS = {  # event id: (day before, day of)
    "20131210": ("2013-12-09", "2013-12-10"),
    "20140125": ("2014-01-24", "2014-01-25"),
    "20131216": ("2013-12-15", "2013-12-16"),
    "20140122": ("2014-01-21", "2014-01-22"),
    "20140214": ("2014-02-13", "2014-02-14"),
}
AREA = [48, -122, 18, -82]
LEVELS = ["1","2","3","5","7","10","20","30","50","70","100","125","150","175","200",
          "225","250","300","350","400","450","500","550","600","650","700","750",
          "775","800","825","850","875","900","925","950","975","1000"]
PL_VARS = ["geopotential", "temperature", "u_component_of_wind",
           "v_component_of_wind", "relative_humidity"]
SL_VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature",
           "2m_dewpoint_temperature", "surface_pressure", "mean_sea_level_pressure",
           "skin_temperature", "sea_surface_temperature", "sea_ice_cover", "snow_depth",
           "land_sea_mask", "geopotential"] \
        + [f"soil_temperature_level_{i}" for i in range(1, 5)] \
        + [f"volumetric_soil_water_layer_{i}" for i in range(1, 5)]

c = cdsapi.Client()
for ev, (d0, d1) in EVENTS.items():
    os.makedirs(f"era5/{ev}", exist_ok=True)
    for day, hours in [(d0, range(18, 24)), (d1, range(0, 19))]:
        y, m, d = day.split("-")
        common = {"product_type": ["reanalysis"], "year": [y], "month": [m], "day": [d],
                  "time": [f"{h:02d}:00" for h in hours], "area": AREA,
                  "data_format": "grib", "download_format": "unarchived"}
        c.retrieve("reanalysis-era5-pressure-levels",
                   {**common, "variable": PL_VARS, "pressure_level": LEVELS},
                   f"era5/{ev}/era5_pl_{y}{m}{d}.grib")
        c.retrieve("reanalysis-era5-single-levels",
                   {**common, "variable": SL_VARS},
                   f"era5/{ev}/era5_sl_{y}{m}{d}.grib")