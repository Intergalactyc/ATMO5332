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

# Steps
Load ERA5 forcing data:
`python era5.py`

Prepare for WPS runs -- creates directory `./wps` with `geo` subdirectory as well as subdirectories for each event (containing namelist.wps and links to executables + ERA5 gribfiles); replace `USER` with username and change locations as necessary:
`python prepare_wps.py ./wps --wps /lustre/work/USER/WPSV4`
- If ERA5 data is located somewhere other than `../era5` (based on project folder), use `--era5` to override.
- If GEOG data is located somewhere other than the template-specified location (should be `/lustre/research/bancell/geogV4/WPS_GEOG`), use `--geog-data-path` to override.
- If this has already been run and you are rerunning to update existing directories, add `--force`

Submit WPS runs: Geogrid, Ungrib, Metgrid. This submits jobs to the slurm queue, by calling sbatch with the submission scripts from `sub` as necessary; it will automatically handle the dependency chain (Geogrid > Ungrib > Metgrid).
`python submit.py wps ./wps`
- If Geogrid has already been run, you can add `--skip-geogrid`
- To see the sbatch commands but not run anything, use `--dry-run`
- Once submit has been run, use `squeue -u USER` to see your jobs in the slurm queue

Prepare for WRF runs -- creates directory `../runs` with subdirectories for each run (event + scheme combination, containing namelist.input, links to executables, tslist, and iofields.txt); replace `USER` with username and change locations as necessary; `{event}` will be replaced by the script with each respective event path, per run:
`python prepare_runs.py ./runs --wrf-run /lustre/work/USER/WRFV4/run --met-em ./wps/{event}`
- For a 3-hr test run: `python prepare_runs.py ./testruns --events 20131210 --schemes ysu`

Submit WRF runs: Real + WRF. This submits jobs to the slurm queue, by calling sbatch with the submission scripts from `sub` as necessary; it will automatically handle the dependency chain (Real > WRF).
`python submit.py wrf ./runs`
- For the test run: `python submit.py wrf ./testruns`
- To see the sbatch commands but not run anything, use `--dry-run`
- Once submit has been run, use `squeue -u USER` to see your jobs in the slurm queue
