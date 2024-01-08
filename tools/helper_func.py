
import os
import numpy as np
import rasterio
from tqdm.auto import tqdm
import h5py



# Set up working directory
if __name__ == '__main__':
    os.chdir('../')



def vrt2tif(vrt_path, tif_path):
    """
    Convert a VRT file to a GeoTIFF file.

    Parameters:
    vrt_path (str): The path to the VRT file.
    dst_path (str): The path to save the GeoTIFF file.

    Returns:
    None
    """
    if os.path.isfile(tif_path):
        print('Skip existing: ', tif_path)
    else:
        # Write the VRT to a TIF
        with rasterio.open(vrt_path) as src:

            profile = src.profile
            # Update the driver to GTiff, and add compression
            profile.update(driver='GTiff',
                           compress='lzw',
                           BIGTIFF = "IF_SAFER")

            with rasterio.open(tif_path, 'w', **profile) as dst:
                # Iterate through blocks
                windows = list(src.block_windows())
                for _, window in tqdm(windows,total=len(windows)):
                    # Read and write the data in each block
                    data = src.read(window=window)
                    dst.write(data, window=window)
        print('File converted successfully!')


        
def tif2hdf(tif_path, chunk_size=None):
    """
    Convert a GeoTIFF file to HDF5 format.

    Args:
        tif_path (str): The path to the GeoTIFF file.

    Returns:
        None
    """
    
    if os.path.isfile(tif_path.replace('.tif','.hdf')):
        print('Skip existing: ', tif_path)
    else:
        # Convert GeoTIFF files to HDF5, keep the chunk size the same as 
        # the raster window size
        with rasterio.open(tif_path,'r') as src:

            # Get the base name of the file
            file_name = os.path.basename(tif_path).split('.')[0]

            # Get raster info
            tif_band_num = src.count
            tif_shape = (tif_band_num,*src.shape)
            tif_dtype = src.dtypes[0]
            # Get block_windows
            block_windows = [window for ij, window in src.block_windows()]
            block_size = (tif_band_num,*src.block_shapes[0])

            # Report the metadata of the file
            print(f'File name: {file_name}')
            print(f'Band number: {tif_band_num}')
            print(f'Raster dtype: {tif_dtype}')
            print(f'Raster shape: {tif_shape}')
            print(f'Block size: {block_size}')
            
            if chunk_size is None:
                chunk_size = block_size
            else:
                chunk_size = (tif_band_num,*chunk_size)
                print(f'Chunk size: {block_size}  =======> {chunk_size}')


            # Create HDF5 file
            with h5py.File(fr'data/raster/{file_name}.hdf','w') as dst:
                
                # Add transform attribute
                dst.create_dataset('transform', data=np.array(list(src.transform)))

                # Create dataset
                dst.create_dataset('array', shape=tif_shape, dtype=tif_dtype, chunks=chunk_size, compression='gzip')

                # Write data to dataset
                for i, window in tqdm(enumerate(block_windows),total=len(block_windows)):

                    # Read data from GeoTIFF
                    data = src.read(window=window)

                    # Write data to HDF5
                    row_slice = slice(window.row_off, window.row_off + window.height)
                    col_slice = slice(window.col_off, window.col_off + window.width)
                    dst['array'][slice(None),row_slice, col_slice] = data

            # Report a successful conversion message
            print(f'File {file_name} converted successfully!')