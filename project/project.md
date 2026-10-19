# Resources
- [TODO list (Google Doc)](https://docs.google.com/document/d/1MaAgE-IjlVOfKC0na4SlL0sTxNxAfBCuPWs6HdiygNU/)
- [ERA5 original paper](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4803)
- ERA5 data access: [single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download), [pressure levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download)
- [era5-for-wrf](https://github.com/moptis/era5-for-wrf/tree/main)
- [WRF Physics Options](https://www2.mmm.ucar.edu/wrf/users/tutorial/presentation_pdfs/201107/WRF_Physics_Dudhia.ppt.pdf)

# Requirements
- Python packages: `cdsapi`, `matplotlib`, `numpy`, `pandas`
- Register/log in to CDS account and [follow the instructions](https://cds.climate.copernicus.eu/how-to-api) to setup an API key (~/.cdsapirc file)
    - Accept the Terms of Use at the bottom of the data access pages ([here](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download) and [here](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download))