# AIxPA-Foreste

This project implements a pipeline for deforestation using Sentinel-2 Level-2A imagery. It processes raw .SAFE or .zip Sentinel-2 inputs, extracts NDVI and BSI indices, interpolates them to a monthly time series, applies BFAST (Breaks For Additive Season and Trend), and outputs change detection and probability maps.

## Input

- **Sentinel-2 L2A Data** in `.SAFE` folders or `.zip` format.
- **Forest Mask** in `.shp` or raster format.
  - Used to limit analysis to forested areas.
  - Can be downloaded from the [WebGIS Portal](https://webgis.provincia.tn.it/) confine del bosco layer or from https://siatservices.provincia.tn.it/idt/vector/p_TN_3d0874bc-7b9e-4c95-b885-0f7c610b08fa.zip.





## Output

- GeoTIFF files for:
- **Change maps** (e.g., `CD_2018_2019.tif`)
- **Post-processed change maps** (e.g., `CD_2018_2019_process.tif`)


These outputs are saved in the specified output directory.

---

## Parameters

The following parameters are required to run the script:

| Parameter  | Description                                    | Example                            |
|------------|------------------------------------------------|------------------------------------|
| `sensor`   | Satellite sensor type (currently only `S2`)    | `'S2'`                              |
| `tilename` | Sentinel-2 tile name                           | `'T32TPS'`                          |
| `years`    | List of years for time series analysis         | `['2018', '2019']`                 |
| `maindir`  | Main directory path for temporary and input data | `'/home/user/'`                  |
| `boscopath`  | Path for forest mask                          | `'/home/user/'`                  |
| `datapath` | Path to the directory containing `.SAFE` data  | `'/path/to/DATA/'`                 |
| `outpath`  | Directory where output files will be saved     | `'/path/to/OUTPUT/'`               |

---

## How It Works

1. **Read Sentinel-2 data** using tile-specific metadata.
2. **Compute NDVI and BSI indices** from RED, NIR, and SWIR1 bands.
3. **Apply cloud/shadow masks** from precomputed binary mask files (`MASK.npy`).
4. **Interpolate data** to generate a complete 24-month time series (12 months/year).
5. **Fuse features** and reshape data into pixel-wise time series.
6. **Run BFAST** to detect change points across time.
7. **Post-process** change maps to remove isolated pixels and fill gaps.
8. **Export results** as GeoTIFF raster files.

---

## Requirements

- Python 3.7+
- Required Python libraries (install via `pip` or `conda`):
- `numpy`
- `joblib`
- `datetime`
- `os`, `shutil`, `json`
- Custom `utils` module (must be included in the repository)

---

## Running the Script

You can run the main script by configuring the parameters and calling the function:

```python
sensor = 'S2'
tilename = 'T32TPS'
years = ['2018','2019']
maindir = '/home/user/'
maskpath = '/home/user/Platform/Bosco'
datapath = '/home/user/Platform/DATA/'
outpath = '/home/user/Platform/OUTPUT/'

deforestation(sensor, tilename, years, maindir, boscopath, datapath, outpath)


