import os
import h5py
import rasterio
from tqdm.auto import tqdm


# Set up working directory
if __name__ == '__main__':
    os.chdir('../')


def GEOTIF2HDF(file):
    """
    Convert GeoTIFF files to HDF5 format.

    Args:
        file (str): The path to the GeoTIFF file.

    Returns:
        None
    """
    # Convert GeoTIFF files to HDF5, keep the chunk size the same as the raster window size
    with rasterio.open(file,'r') as src:

        # Get the base name of the file
        file_name = os.path.basename(file).split('.')[0]

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


        # Create HDF5 file
        with h5py.File(fr'data/raster/{file_name}.hdf5','w') as dst:

            # Create dataset
            dst.create_dataset('array', shape=tif_shape, dtype=tif_dtype, chunks=block_size, compression='gzip')

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