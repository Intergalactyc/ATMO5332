# Information
**Overarching question**: *Which PBL parameterization schemes best reproduce the nocturnal stable boundary layer and associated low-level jet over the South Plains region?*

This project involves running WRF-ARW v4 on several days with known nocturnal low-level jets (LLJs) with noses below 200 meters (as observed by the Texas Tech University 200-meter tower at Reese Technology Center). Several PBL closure schemes will be compared for each event. Initial and LBC forcing data comes from ERA5 reanalysis. Analysis of results will compare model outputs to surface observations made by the West Texas Mesonet, as well as to observations from TTU's 200-m tower, especially for flux comparisons and study of the LLJ wind profile.

Schemes being compared:
- Yonsei University Scheme (YSU)
    - bl_pbl_physics=1
    - 1st-order non-local mixing closure
- Mellor-Yamada-Nakanishi-Niino Level 2.5 (MYNN)
    - bl_pbl_physics=5
    - Also known as MYNN2 or MYNN2.5 (in contrast to the Level 2.6 MYNN26/MYNN2.6 or the Level 3 MYNN3)
    - 1.5-order local TKE closure
- Asymmetric Convective Model version 2 (ACM2)
    - bl_pbl_physics=7
    - 1st-order hybrid closure

All schemes use Jimenez's Revised MM5 Monin-Obukhov surface-layer scheme (sf_sfclay_physics=1). MYNN has a paired MYNN surface-layer scheme (sf_sfclay_physics=5), but that is not used here; instead the same surface-layer option is used for all three PBL schemes, allowing a more fair comparison of the PBL schemes themselves.

MYJ (bl_pbl_physics=2) was also considered as a PBL scheme option, but as it is relatively similar to MYNN, and also requires a slightly different Monin-Obukhov-based surface layer scheme (sf_sfclay_physics=2), it has been left out for simplicity.

LLJ event dates (early morning each day):
- 10 December 2013 (*)
- 16 December 2013
- 22 January 2014
- 25 January 2014 (*)
- 14 February 2014

Time period (per event) runs from 18 UTC the day prior to 18 UTC the day of. This is the period for which data is acquired (IC/LBC ERA5 data for WRF, as well as mesonet + tower data for analysis) and also the period over which the model is run.

Spatial domains:
- d01: 9 km, 2250 x 1980 km size, covering appx. 24.2° to 42.5° lat, -115.6° to -88.6° long
    - This covers around 1/3 of the US (~South Central): all of TX, northern Mexico, south Rockies, NW Gulf; roughly centered around d02
    - Purpose: regional synoptic forcing, including effects from Rockies in the West and Gulf moisture in the SE
- d02: 3 km, 837 x 810 km size, covering appx. 29.9° to 37.3° lat, -106.6° to -97.2° long
    - Covers Eastern NM, west TX, west OK; roughly centered around d03
    - Purpose: regional convection; intermediate grid for transition from 9 km down to 1 km
- d03: 1 km, 366 x 366 km size, covering appx. 31.9° to 35.3° lat, -103.9° to -99.9° long
    - Covers the South Plains region; centered near Lubbock
    - Purpose: high-resolution model of the South Plains ROI

The ROI for analysis is 32.10° to 35.10° latitude, -103.67° to -100.11° longitude. These bounds were chosen to remain at least 15 km within d03 edge (to avoid edge spinup issues). Analysis should use all and only mesonet stations within this region (tslist is generated to use specifically these stations).


# Requirements
- Python packages: `cdsapi`, `matplotlib`, `numpy`, `pandas`, `netcdf4`
    - Have the `ncl_stable` environment properly set up and activated
    - Use `conda install -c conda-forge <packages>` to install packages as needed
- Register/log in to CDS account and [follow the instructions](https://cds.climate.copernicus.eu/how-to-api) to setup an API key (~/.cdsapirc file)
    - Accept the Terms of Use at the bottom of the data access pages ([here](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download) and [here](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download))

# Steps to run on HPCC
**Load ERA5 forcing data**:

```python era5.py```

**Prepare for WPS runs.** This creates a directory `./wps` with `geo` subdirectory as well as subdirectories for each event (containing namelist.wps and links to executables + ERA5 gribfiles). Replace `USER` with username and change locations as necessary.

```python prepare_wps.py ./wps --wps /lustre/work/USER/WPSV4```
- If ERA5 data is located somewhere other than `../era5` (based on project folder), use `--era5` to override.
- If GEOG data is located somewhere other than the template-specified location (should be `/lustre/research/bancell/geogV4/WPS_GEOG`), use `--geog-data-path` to override.
- If this has already been run and you are rerunning to update existing directories, add `--force`

**Submit WPS runs: Geogrid, Ungrib, Metgrid.** This submits jobs to the slurm queue, by calling sbatch with the submission scripts from `sub` as necessary; it will automatically handle the dependency chain (Geogrid > Ungrib > Metgrid).

```python submit.py wps ./wps```
- If Geogrid has already been run, you can add `--skip-geogrid`
- To see the sbatch commands but not run anything, use `--dry-run`
- Once submit has been run, use `squeue -u USER` to see your jobs in the slurm queue. If a job needs to be canceled, use `scancel JOBID`.

**Prepare for WRF runs.** This creates a directory `../runs` with subdirectories for each run (event + scheme combination, containing namelist.input, links to executables, tslist, and iofields.txt). Replace `USER` with username and change locations as necessary. `{event}` will be replaced by the script with each respective event path, per run.

```python prepare_runs.py ./runs --wrf-run /lustre/work/USER/WRFV4/run --met-em ./wps/{event}```
- For a single 3-hr test run: `python prepare_runs.py ./testruns --events 20131210 --schemes ysu --run-hours 3 --wrf-run /lustre/work/USER/WRFV4/run --met-em ./wps/{event}`

**Submit WRF runs: Real + WRF.** This submits jobs to the slurm queue, by calling sbatch with the submission scripts from `sub` as necessary; it will automatically handle the dependency chain (Real > WRF).

```python submit.py wrf ./runs```
- For the test run: `python submit.py wrf ./testruns`
- To see the sbatch commands but not run anything, use `--dry-run`
- Once submit has been run, use `squeue -u USER` to see your jobs in the slurm queue. If a job needs to be canceled, use `scancel JOBID`.

**Pack and transfer results.**
Pack everything (station time series, logs, and the tower column extracted from the d03 wrfout files) into one archive, as a batch job:

```
sbatch -p nocona -N 1 --ntasks-per-node=16 -t 02:00:00 -o pack-%j.log --wrap "python3 pack_results.py ./runs --workers 16"
```

Finally, use SCP or similar to transfer the archive to your local computer for analysis, and extract the archive. For example:

```
scp username@login.hpcc.ttu.edu:/path/to/ATMO5332/project/results.tar.gz .
mkdir runs
tar -xzf results.tar.gz -C runs
```

Wrfout files themselves are not packed or transferred in the above.

# Resources
- [ERA5 original paper](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4803)
- ERA5 data access: [single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download), [pressure levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download)
- [WRF Physics User Guide](https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/physics.html)
- [WRF Physics Options](https://www2.mmm.ucar.edu/wrf/users/tutorial/presentation_pdfs/201107/WRF_Physics_Dudhia.ppt.pdf)
- Stensrud's *Parameterization Schemes* (Ch. 5) is very useful

# Relevant papers
- Storm, B., Dudhia, J., Basu, S., Swift, A. and Giammanco, I. (2009), Evaluation of the Weather Research and Forecasting model on forecasting low-level jets: implications for wind energy. Wind Energ., 12: 81-90. https://doi.org/10.1002/we.288.
- Draxl, C., Hahmann, A.N., Peña, A. and Giebel, G. (2014), Evaluating winds and vertical wind shear from Weather Research and Forecasting model forecasts using seven planetary boundary layer schemes. Wind Energ., 17: 39-55. https://doi.org/10.1002/we.1555.
- Mirocha, J. D., Simpson, M. D., Fast, J. D., Berg, L. K., and Baskett, R. L. (2016) Investigation of boundary-layer wind predictions during nocturnal low-level jet events using the Weather Research and Forecasting model. Wind Energ., 19: 739–762. https://doi.org/10.1002/we.1862.
- Xie, B., J. C. H. Fung, A. Chan, and A. Lau (2012), Evaluation of nonlocal and local planetary boundary layer schemes in the WRF model, J. Geophys. Res., 117, D12103, https://doi.org/10.1029/2011JD017080.
- Shin, H.H., Hong, SY. Intercomparison of Planetary Boundary-Layer Parametrizations in the WRF Model for a Single Day from CASES-99. Boundary-Layer Meteorol 139, 261–281 (2011). https://doi.org/10.1007/s10546-010-9583-z.
