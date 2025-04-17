import sys
import numpy as np
import sys, os, time, shutil, json
from os.path import abspath
import utils.filemanager as fm
from utils.S2L2A import L2Atile, getTileList
from utils.utils import _ndi, _bsi
from datetime import datetime
from joblib import Parallel, cpu_count, delayed
from utils.utils import run_bfast_parallel, get_month_numbers, interpolate_for_year, interpolate_time_series, fuse_features, parallel_interpolate
import utils.post_processing as pp
import utils.custom_bfast as bfast
from tqdm import tqdm


def deforestation(sensor, tilename, years, maindir, boscopath, datapath, outpath):
    # Initialize logging and timer
    logging = {} 
    t_tot = time.time()



    # Check sensor type and get tile list
    if sensor == 'S2':
        tiledict = getTileList(datapath)
    else:
        raise IOError('Invalid sensor')
    
    keys = tiledict.keys()

    for k in keys:
        tileDatapath = tiledict[k]
        print(f"Reading Tile-{k}.")
        
        if sensor == 'S2':
            tile = L2Atile(maindir, tileDatapath)

        # Initialize empty storage for all years
        
        feature_file = os.path.join(outpath, 'feature_all.dat')

        timestep_index = 0  # to keep track of writing index

        all_dates = []    

        for y in years:
            # Set temporary path for the current year
            temppath = fm.joinpath(maindir, 'numpy', tilename)

            # Get features for the current year
            ts, _, _ = tile.gettimeseries(year=y, option='default')
            fn = [f for f in os.listdir(temppath)] 

            if len(ts) != 0:
                print(f'Extracting features for each image for year {y}:')
            
            # Get some information from data
            height, width = ts[0].feature('B04').shape
            geotransform, projection = fm.getGeoTIFFmeta(ts[0].featurepath()['B04'])
            ts_length = len(ts)

        
           

            if timestep_index == 0:
                # Initialize memory-mapped arrays with estimated total time steps
                n_timesteps_total = sum([len(tile.gettimeseries(year=y, option='default')[0]) for y in years])
                feature_all = np.memmap(feature_file, dtype='float16', mode='w+', shape=(height, width, n_timesteps_total))
                #read bosco map
                geotransform, projection = fm.getGeoTIFFmeta(ts[0].featurepath()['B04'])
                bosco_mask = fm.shapefile_to_array(boscopath, geotransform, projection, height, width, attribute='objectid')
            

            ts = sorted(ts, key=lambda x: x.InvalidPixNum())[0:ts_length]
            totimg = len(ts)

            dates = []
               
            # Compute Index Statistics
            for idx, img in enumerate(ts):        
                print(f'.. {idx+1}/{totimg}      ', end='\r')   
                        
                # Compute NDVI and BSI indices
                b1 = img.feature('BLUE', dtype=np.float16)
                b3 = img.feature('RED', dtype=np.float16)
                b4 = img.feature('nir', dtype=np.float16)
                b5 = img.feature('SWIR1', dtype=np.float16)
                
                
    
                NDVI = _ndi(b4, b3)
                BSI = _ndi(b1, b3, b4, b5)

                fuse_feature = fuse_features(NDVI,BSI)
    
                # Mask for valid values (update if needed)
                fn = fn[1:]
                name = fn[idx]
                maskpath = fm.joinpath(temppath, name, 'MASK.npy')
                msk = np.load(maskpath)

                feature_mask = (np.where(msk, np.nan, fuse_feature))
                

                feature_all[:, :, timestep_index] = feature_mask


                all_dates.append(img._metadata['date'])
                timestep_index += 1
                

                # Delete intermediate arrays to free memory
                del b3, b4, b5, NDVI, BSI, msk, fuse_feature, feature_mask


                
        # Flush memory-mapped arrays to disk
        feature_all.flush()
        

    
    #read the dates
    # Convert the date strings to datetime objects
    all_dates_datetime = [datetime.strptime(date, '%Y%m%d') for date in all_dates]
    
    # Separate dates based on the year
    dates_2018 = [date for date in all_dates_datetime if date.year == 2018]
    dates_2019 = [date for date in all_dates_datetime if date.year == 2019]

    
    #feature data
    feature_data = feature_all
    
    #filter by bosco map
    feature_data = np.where(bosco_mask[:,:,np.newaxis] == 0, np.nan, feature_data).astype(np.float16)
    height, width, time_steps = feature_data.shape
    
    interpolated_feature = np.zeros((height, width, 24), dtype=np.float16)

    # Flatten image and get valid pixel indices (not NaN across all time steps)
    flat_pixels = feature_data.reshape(-1, time_steps)
    valid_mask = ~np.isnan(flat_pixels).all(axis=1)
    valid_pixels = flat_pixels[valid_mask]
    
    print(f"Total pixels: {flat_pixels.shape[0]}, Valid pixels: {valid_pixels.shape[0]}")
    

    # Vectorized approach for interpolation
    print('Generating monthly samples:')
        interpolated_valid = Parallel(n_jobs=-1)(
        delayed(interpolate_time_series)(px, dates_2018, dates_2019)
        for px in tqdm(valid_pixels, desc="Interpolating")
    )
    interpolated_valid = np.stack(interpolated_valid).astype(np.float16)
    #interpolated_feature = np.apply_along_axis(interpolate_time_series, 2, feature_data, dates_2018, dates_2019)
    #interpolated_feature = parallel_interpolate(feature_data, dates_2018, dates_2019)


    # Reshape for BFAST
    #totpixels = height * width
    #fused_reshaped = interpolated_feature.reshape((totpixels, 24))
   
    
    # Run BFAST
    print('Running break point detector:')
    tot_valid = interpolated_valid.shape[0]
    startyear = int(years[0])
    endyear = int(years[-1]) 
    freq = 12 #monthly data
    nyear = endyear - startyear 
    years_np = np.arange(startyear, endyear+1)
    
    
    with Parallel(n_jobs=-1) as parallel:
        dates = bfast.r_style_interval((startyear, 1), (startyear + nyear, 365), freq).reshape(interpolated_valid.shape[1], 1)
        breaks, confidence = run_bfast_parallel(parallel, interpolated_valid, dates, freq)
          
    # Process results
    changemaps = breaks // freq
    accuracymaps = confidence
    changemaps = changemaps.reshape(height, width)
    accuracymaps = accuracymaps.reshape(height, width)
    
    
    
    changemaps_year = np.zeros_like(changemaps, dtype = int)
    for i, year in enumerate(years_np):
        changemaps_year[changemaps == i] = year


    # Initialize full-size output arrays with a fill value (e.g., 0 or np.nan)
    full_changemap = np.full((height * width,), 0, dtype=int)
    full_confidence = np.full((height * width,), 0, dtype=np.float16)
    
    # Put results back into full-size arrays
    full_changemap[valid_mask] = changemaps_year
    full_confidence[valid_mask] = confidence
    
    # Reshape to 2D maps
    changemaps_final = full_changemap.reshape(height, width)
    confidence_final = full_confidence.reshape(height, width) 
    
    
    # Remove isolated pixels
    updated_change_array, updated_probability_array = pp.remove_isolated_pixels(changemaps_final, confidence_final)
    
    # Fill gaps and update probabilities
    final_change_array, final_probability_array = pp.fill_small_holes_and_update_probabilities(updated_change_array, updated_probability_array) 
    
    final_change_array = final_change_array.astype(float)
    final_probability_array = final_probability_array.astype(float)
    final_change_array[final_change_array ==0 ] = np.nan
    final_probability_array[final_probability_array ==0 ] = np.nan
    
    
    # Save output
                
    output_filename_process = fm.joinpath(outpath,"CD_2018_2019")
    fm.writeGeoTIFFD(output_filename_process, np.stack([final_change_array, final_probability_array], axis=-1), geotransform, projection) 
                   
    
    
    print("Processing complete!")    
         

#PREPARE SOME TOOLBOX PARAMETERS
sensor = 'S2'
tilename = 'T32TPS'
years = ['2018','2019']
maindir = '/home/user/'
boscopath = '/home/user/Bosco/'
datapath = '/home/user/DATA/'
outpath = '/home/user/OUTPUT/'
temppath = fm.joinpath(maindir, 'numpy')

deforestation(sensor, tilename, years, maindir, boscopath, datapath, outpath)
