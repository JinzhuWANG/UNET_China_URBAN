from glob import glob
import rasterio
import numpy as np
import h5py
from tqdm.auto import tqdm


# Get all GeoTIFF files 
files = glob('data/raster/*.tif')
file = 'data/raster/CLCD_v01_2006_albert.tif'

# Convert GeoTIFF files to HDF5, keep the chunk size the same as the raster window size
with rasterio.open(file,'r') as src:
    
    # Get the base name of the file
    file_name = file.split('/')[-1].split('.')[0]
    
    # Get raster info
    tif_band_num = src.count
    tif_shape = (tif_band_num,*src.shape)
    tif_dtype = src.dtypes[0]
    # Get block_windows
    block_windows = [window for ij, window in src.block_windows()]
    block_size = (tif_band_num,*src.block_shapes[0])
    
    # Create HDF5 file
    with h5py.File(f'{file_name}.hdf5','w') as dst:
        
        # Create dataset
        dst.create_dataset('array', shape=tif_shape, dtype=tif_dtype, chunks=block_size, compression='gzip')
        
        # Write data to dataset
        for i, window in tqdm(enumerate(block_windows)):
            
            # Read data from GeoTIFF
            data = src.read(window=window)
            
            # Write data to HDF5
            row_slice = slice(window.row_off, window.row_off + window.height)
            col_slice = slice(window.col_off, window.col_off + window.width)
            dst['array'][row_slice, col_slice] = data
            
    