import numpy as np
import re
from datetime import datetime
from joblib import Parallel, cpu_count, delayed
#import bfast
import filemanager as fm
import custom_bfast as bfast


###Some Functions
def _ndi(b1,b2):

    denom = b1 + b2        
    nom = (b1-b2)

    denom[denom==0] = 1e-8
    index = nom/denom  

    index[index>1] = 1
    index[index<-1] = -1
    
    return index 

# Map dates to month numbers
def get_month_numbers(dates):
    return np.array([d.month for d in dates])

# Interpolate data for a single year
def interpolate_for_year(pixel_data, dates):
    month_numbers = get_month_numbers(dates)
    valid_indices = ~np.isnan(pixel_data)
    valid_months = month_numbers[valid_indices]
    valid_data = pixel_data[valid_indices]
    target_months = np.arange(1, 13)
    if len(valid_data) == 0:
        return np.zeros(12)
    return np.interp(target_months, valid_months, valid_data).astype(np.float16)

# Interpolate data for both years
def interpolate_time_series(pixel_data, dates_2018, dates_2019):
    pixel_data_2018 = pixel_data[:len(dates_2018)]
    pixel_data_2019 = pixel_data[len(dates_2018):]
    interpolated_2018 = interpolate_for_year(pixel_data_2018, dates_2018)
    interpolated_2019 = interpolate_for_year(pixel_data_2019, dates_2019)
    return np.concatenate([interpolated_2018, interpolated_2019]).astype(np.float16)

# Fusion of NDVI and BSI
def fuse_features(ndvi, bsi):
    return np.sqrt((ndvi ** 2 + bsi ** 2) / 2).astype(np.float16)
    
    
# Fusion parallel processing    
def process_pixel(i, j):
    ndvi_pixel = ndvi_data[i, j, :]
    bsi_pixel = bsi_data[i, j, :]
    interpolated_ndvi = interpolate_time_series(ndvi_pixel, dates_2018, dates_2019)
    interpolated_bsi = interpolate_time_series(bsi_pixel, dates_2018, dates_2019)
    return i, j, fuse_features(interpolated_ndvi, interpolated_bsi)
    

# Parallel BFAST processing
def run_bfast_parallel(par_mngr, ts_2D, dates, freq, verbosity=0):
    step = max(len(ts_2D) // cpu_count(), 100)
    parallel_range = range(0, len(ts_2D), step)
    results = list(
        zip(
            *par_mngr(
                delayed(bfast.bfast_cci)(  # Call BFAST analysis
                    ts_2D[start: start + step].T,
                    dates,
                    h=(freq / 2) / (ts_2D.shape[1]),
                    verbosity=verbosity,
                )
                for start in parallel_range
            )
        )
    )
    return np.concatenate(results[0], axis=0), np.concatenate(results[1], axis=0)
                   