# AIxPA-Foreste

This project implements a pipeline for deforestation using Sentinel-2 Level-2A imagery. It processes raw .SAFE or .zip Sentinel-2 inputs, extracts NDVI and BSI indices, interpolates them to a monthly time series, applies BFAST (Breaks For Additive Season and Trend), and outputs change detection and confidence maps.

🚀 Overview
The main script performs the following steps:

Reads and processes Sentinel-2 Level-2A data for a specified tile and years.

Computes NDVI and BSI indices for each image.

Masks invalid pixels using external masks.

Interpolates the indices to obtain a complete monthly time series.

Fuses NDVI and BSI features for each pixel.

Applies BFAST to detect changes.

Post-processes the change maps to remove noise and fill holes.

Saves final shapefiles representing change year and probability.

📥 Input
Sentinel-2 data in .SAFE folder format or .zip archives.

External mask files (in .npy format) for masking invalid pixels.

Data should be organized by tile in the following structure:

markdown
Copy
Edit
DATA/
  └── T32TPS/
      ├── S2A_MSIL2A_2018...SAFE
      ├── S2B_MSIL2A_2018...SAFE
      └── ...
📤 Output
Shapefiles will be generated and saved in the specified output directory:

CD_2018_2019.shp – Raw BFAST change map with year of detected change.

prob_2018_2019.shp – Raw confidence map (BFAST magnitude).

CD_2018_2019_process.shp – Processed change map with noise removed.

prob_2018_2019_process.shp – Processed confidence map.

All outputs are GeoTIFF-compatible files, georeferenced with metadata extracted from the original Sentinel-2 scenes.

⚙️ Parameters
The following parameters are defined in the main script:

python
Copy
Edit
sensor = 'S2'  # Sentinel-2 sensor
tilename = 'T32TPS'  # Sentinel-2 tile ID
years = ['2018', '2019']  # Years to process
maindir = '/home/username/'  # Main working directory
datapath = '/path/to/DATA/'  # Input directory containing Sentinel-2 .SAFE or .zip data
outpath = '/path/to/OUTPUT/'  # Output directory for saving results
Update the paths to match your local directory structure.

🧩 Dependencies
Make sure the following Python libraries are installed:

numpy

joblib

datetime

shutil

os, sys, time

Custom utility scripts in utils/:

filemanager.py

S2L2A.py

utils.py

post_processing.py

🛠️ Usage
Run the pipeline directly from the terminal or inside your Python environment:

bash
Copy
Edit
python main.py
Make sure all required utils/ scripts are accessible in the same directory or Python path.

📝 Notes
This pipeline assumes pre-existing .npy mask files for each image.

The script is parallelized for performance using all available CPU cores.

BFAST breaks are reported in monthly resolution and later converted to corresponding year.

📬 Contact
For issues, questions, or suggestions, feel free to open an issue or contact the project maintainer.
