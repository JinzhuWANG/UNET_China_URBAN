from glob import glob
import numpy as np

from tools import GEOTIF2HDF


# Get all GeoTIFF files 
files = glob('data/raster/*.tif')

# Convert GeoTIFF files to HDF5
for file in files:            
    GEOTIF2HDF(file)    