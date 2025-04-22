import os, errno, pickle, gc
from scipy import io, misc
from scipy.signal import butter, lfilter, iirnotch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors, figure
import datetime
from skimage import transform
import geopandas as gpd
from shapely.geometry import Polygon
from osgeo import gdal, ogr, osr


#--------------------------------------------------------#
# BASIC I/O UTILITY

def check_folder(*paths):
    path = joinpath(*paths)
    #if folder doesn't exist..
    if not os.path.exists(path):
        #..try creating it
        try:
            os.makedirs(path)
        #..else raise exception
        except OSError as e:
            if (e.errno != errno.EEXIST):
                raise   
    return path

def formatPath(path):
    newpath = os.path.normpath(path)
    return newpath

def joinpath(*argv):
    fp = ''
    for arg in argv:
        fp += str(arg) + '/'
    return formatPath(fp)

def savevar(varpath, var):
    #PREPARE SAVEPATH
    if varpath.endswith('.pkl'):
        savepath = varpath 
    else:
        savepath = varpath + '.pkl'

    #STORE VARIABLE
    f = open(savepath, 'wb')
    pickle.dump(var, f, 3) #python3 compatible protocol
    f.close()

def loadvar(varpath):
    #PREPARE LOADPATH
    if varpath.endswith('.pkl'):
        loadpath = varpath 
    else:
        loadpath = varpath + '.pkl'

    if os.path.isfile(loadpath):
        #LOAD VARIABLE
        f = open(loadpath, 'rb')
        var = pickle.load(f) 
        f.close()
        return var
    else:
        raise IOError('File does not exist!')

def savemat(var, savepath, name):
    if name.endswith('.mat'):
        fn = name 
    else:
        fn = name + '.mat'
    fp = joinpath(savepath, fn)
    matdict = {fn[:-4]:var}
    io.savemat(fp,matdict)

def loadmat(loadpath, name):
    if name.endswith('.mat'):
        fn = name 
    else:
        fn = name + '.mat'
    fp = joinpath(loadpath, fn)
    matdict = io.loadmat(fp)
    var = matdict[fn[:-4]]

    return var

def imsave(savepath, matr, **kwargs):
    colormap = kwargs.get('colormap','gnuplot')
    vmin = kwargs.get('vmin',0)
    vmax = kwargs.get('vmax',np.amax(matr))

    if savepath.endswith('.png'):
        fn = savepath 
    else:
        fn = savepath + '.png'

    my_cmap = cm.get_cmap(colormap)  
    my_cmap.set_under('w')
    plt.imsave(fn,matr, cmap=my_cmap, vmin=vmin, vmax=vmax)   
    return None

"""
def imread(filepath):
    matr = misc.imread(filepath)
    return matr
"""
def array_as_image(array, path, name='image.png', **kwargs):
    colormap = kwargs.get('colormap','gnuplot')
    title = kwargs.get('title',None)
    vmin = kwargs.get('vmin',0)
    vmax = kwargs.get('vmax',1)
    cbar_lsize = kwargs.get('labelsize',30)
    
    fig = plt.figure( figsize=(3840/100,2160/100) )  
    my_cmap = cm.get_cmap(colormap)  
    my_cmap.set_under('w')
    my_norm = colors.Normalize(vmin=vmin, vmax=vmax)
    img = plt.imshow(array, cmap=my_cmap, norm=my_norm)
    img.axes.get_xaxis().set_visible(False)
    img.axes.get_yaxis().set_visible(False)
    cbar = plt.colorbar(img)
    cbar.ax.tick_params(labelsize=cbar_lsize) 
    if title:
        cbar.set_label(title,size=cbar_lsize)
    sp = joinpath(path, name)
    fig.savefig(sp)
    plt.close(fig)
    img,fig = None,None    

#--------------------------------------------------------#
def readGeoTIFF(path, metadata=False):
    """If metadata=False(default) returns array;
    else returns in the following order:
    -array
    -geotransform=(Ix(0,0), res(W-E), 0, Iy(0,0), -res(N-S))
    -projection
    """

    gobj = gdal.Open(path, gdal.GA_ReadOnly)
    if gobj:
        raster = gobj.GetRasterBand(1)
        matr = raster.ReadAsArray()
        geotransform = gobj.GetGeoTransform()
        projection = gobj.GetProjection() 
        if (metadata==True):
            return matr, geotransform, projection
        else:
            return matr
    else:
        raise Exception('Reading Failure: GDALOpen() returned None!')
    gobj = None
    return matr

def getGeoTIFFmeta(filepath):
    """Returns in the following order:
    -geotransform=(Ix(0,0), res(W-E), 0, Iy(0,0), -res(N-S))
    -projection
    """
    gobj = gdal.Open(filepath, gdal.GA_ReadOnly)
    if gobj:        
        geotransform = gobj.GetGeoTransform()
        projection = gobj.GetProjection()

        gobj = None
        return geotransform, projection
    else:
        Exception('Reading Failure: GDALOpen() returned None!')  
        


def write_shapefile(array, transform, crs, output_shapefile):
    """
    Converts a NumPy array to a shapefile using GDAL.
    
    Parameters:
        array (numpy.ndarray): Input raster data.
        transform (tuple): Affine transformation (GeoTransform).
        crs (str or int): Coordinate reference system (EPSG code or WKT).
        output_shapefile (str): Path to save the output shapefile.
    """
    # Create a memory raster dataset
    driver = gdal.GetDriverByName("MEM")
    rows, cols = array.shape
    mem_raster = driver.Create("", cols, rows, 1, gdal.GDT_Int32)

    # Set geotransform and projection
    mem_raster.SetGeoTransform(transform)
    srs = osr.SpatialReference()
    
    if isinstance(crs, int):  # If EPSG code is given
        srs.ImportFromEPSG(crs)
    else:  # If WKT is given
        srs.ImportFromWkt(crs)

    mem_raster.SetProjection(srs.ExportToWkt())

    # Write array to band
    band = mem_raster.GetRasterBand(1)
    band.WriteArray(array)

    # Create shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if driver is None:
        raise RuntimeError("Shapefile driver not available.")

    # Remove existing file if necessary
    if output_shapefile:
        driver.DeleteDataSource(output_shapefile)

    # Create shapefile dataset
    shp_ds = driver.CreateDataSource(output_shapefile)
    shp_layer = shp_ds.CreateLayer("features", srs, ogr.wkbPolygon)

    # Add attribute field
    field_defn = ogr.FieldDefn("Value", ogr.OFTInteger)
    shp_layer.CreateField(field_defn)

    # Polygonize the raster
    gdal.Polygonize(band, None, shp_layer, 0, [], callback=None)

    # Cleanup
    shp_ds = None
    mem_raster = None

    print(f"Shapefile saved to: {output_shapefile}")

        
"""
def readGeoTIFFraster(path):
    with rasterio.open(path) as dataset:
        # Read the data (raster values)
        data = dataset.read(1)  # Read the first band (index 1)
        
        # Get the geotransform
        geotransform = dataset.transform
        
        # Get the projection (CRS)
        projection = dataset.crs
        
        return data, geotransform, projection

def write_shapefile(array, transform, crs, output_shapefile):

    # Convert array to vector shapes (polygons)
    shapes_generator = shapes(array, transform=transform)

    polygons = []
    values = []

    for geom, value in shapes_generator:
        if value != 0:  # Filter out background (0) if necessary
            polygons.append(Polygon(geom['coordinates'][0]))
            values.append(value)

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame({"value": values}, geometry=polygons, crs=crs)

    # Save to a shapefile
    gdf.to_file(output_shapefile, driver="ESRI Shapefile")   
"""    
#--------------------------------------------------------#
# ARRAY PROCESSING

def rescale(matrix, scale, interpolation_type='bilinear'):
    """
    https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.rescale
    https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.warp
    """
    #GET SOME INFORMATION
    interp = {
        'nearest': 0, #for lazy people :D
        'nearestneighbor': 0,
        'nearest_neighbor':0,
        'bilinear': 1,
        'bicubic':3
    }
    if interpolation_type in interp.keys():
        interpolation = interp[interpolation_type]
    else:
        raise Exception('Provided interpolation type is not valid!')
    multich = ( len(matrix.shape)>2 )
    antialias = (scale<1) 
    datatype = matrix.dtype

    #SOME USEFUL WARNINGS
    if (interpolation=='bilinear') & (scale>1):
        raise Warning('When upscaling, "Bicubic" is suggested!')  
    elif (interpolation=='bicubic') & (scale<1):
        raise Warning('When downscaling, "Bilinear" is suggested!')        
    
    #RESCALE
    matr = transform.rescale(matrix, scale, 
                    mode='reflect', 
                    order = interpolation,  
                    anti_aliasing=antialias, 
                    preserve_range=True)

    return matr.astype(datatype)
#--------------------------------------------------------#
# MANAGE DATES
def string2ordinal(string, fmt="%Y%m%d"):
    """ INPUT: string=string to convert into ordinal day; fmt = format of the string."""
    d = datetime.datetime.strptime(string, fmt).toordinal()
    return d

def ordinal2string(num, fmt="%Y%m%d"):
    d = datetime.datetime.fromordinal(num).strftime(fmt)
    return d

#--------------------------------------------------------#
# DISPLAY IMAGES

def imshow(*images, share=True):
    totimg = len(images)
    rows = 1
    cols = 1
    
    while ((rows*cols)<totimg ):
        cols +=1
        if ((rows*cols)<totimg ):
            rows +=1
    
    f, _ = plt.subplots(nrows=rows, ncols=cols, sharey=share, sharex=share)
    for idx,img in enumerate(images):
        xs = f.axes[idx]
        xs.imshow(img)

    if (len(f.axes)>(idx+1)):
        for jdx in range((idx+1),len(f.axes)):
            f.delaxes(f.axes[(jdx)])
    plt.show()

def plot(*functions):
    for idx, f in enumerate(functions):
        if ( len(f)==2 ):
            x = f[0]
            y = f[1]
            lbl= str(idx+1)
            plt.plot(x,y, label=lbl)
        else:
            y = f
            lbl= str(idx+1)
            plt.plot(y, label=lbl)
    plt.legend()
    plt.show()

def imshow3D(matr):
    from mpl_toolkits.mplot3d import Axes3D
    xx, yy = np.mgrid[0:matr.shape[0], 0:matr.shape[1]]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, matr ,rstride=1, cstride=1, cmap='jet',linewidth=2)
    ax.view_init(80, 30)
    plt.show()


#--------------------------------------------------------#
# SIGNAL PROCESSING

def fft(signal, coupled='DC', show=False):
    if coupled=='AC':
        mean = np.mean(signal)
        s = signal-mean
        fft_signal = np.fft.fft(s)
    if coupled=='DC':
        fft_signal = np.fft.fft(signal)

    if show==True:
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.title.set_text('Signal')
        ax2.title.set_text('FFT')
        ax1.plot(signal)
        ax2.plot(fft_signal)
        plt.show()

    return fft_signal

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')   
    y = lfilter(b, a, data)
    return y

def notch_filter(data, cutfreq, fs, quality=1):
    b, a = iirnotch(cutfreq, quality, fs)
    y = lfilter(b, a, data)
    return y

def bandstop_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    y = lfilter(b, a, data)
    return y

def var_local(img,win_size=3):
    from scipy.ndimage import generic_filter
    
    var = generic_filter(img, np.var, size=win_size)    
    return var

def mean_local(img,win_size=3):
    from scipy.ndimage import generic_filter
    
    var = generic_filter(img, np.mean, size=win_size)    
    return var
