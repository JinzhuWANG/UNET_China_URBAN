import os
import geopandas as gpd
import h5py
import numpy as np
from osgeo import gdal
import pyproj
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from shapely.geometry import box
from tqdm.auto import tqdm




# Set up working directory
if __name__ == '__main__':
    os.chdir('../')


def GEOTIF2HDF(files):
    """
    Convert GeoTIFF files to HDF5 format.

    Args:
        files (list): List of file paths to GeoTIFF files.

    Returns:
        None
    """
    for file in files:
        # skip if the file is already converted
        if os.path.isfile(file.replace('.tif', '.hdf')):
            print('Skip existing: ', file)
        else:
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
                with h5py.File(fr'data/raster/{file_name}.hdf','w') as dst:

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


def remove_out_bounds_pts(ref_tif, ROI_box):
    """
    Remove points from the ROI box that are outside the bounds of the reference raster.

    Args:
        ref_tif (str): Filepath of the reference raster.
        ROI_box (str): Filepath of the ROI box.

    Returns:
        geopandas.GeoDataFrame: ROI box with points removed that are outside the bounds of the reference raster.
    """
    # get the CRS of the reference raster
    raster_src = rasterio.open(ref_tif)
    crs = pyproj.CRS(raster_src.crs)

    # Get the top/bottom/left/right coordinates of the reference raster
    left, bottom, right, top = raster_src.bounds

    # Subtract right/bottom by the block size, so that the sample points (coor:coor + block_shape) 
    # will not pass the edge
    right = right - (raster_src.block_shapes[0][1]/2)
    bottom = bottom + (raster_src.block_shapes[0][0]/2)

    sample_box = box(left, bottom, right, top)

    # Read the ROI box, and convert it to the same CRS as the reference raster
    sample_ROI = gpd.read_file(ROI_box)
    sample_ROI = sample_ROI.to_crs(crs)

    # Add a centroid column
    sample_ROI['centroid'] = sample_ROI['geometry'].centroid

    # Check if the centroid is within the bounding box of the reference raster
    sample_ROI = sample_ROI[sample_ROI['centroid'].within(sample_box)]

    return sample_ROI[['centroid']]


# Create en empty destination raster with HDF format
def create_empty_raster(src_file, dst_file, dst_crs, resolution=30, chunk_size=512):
    """
    Create an empty raster HDF file with the same shape and dtype as the source raster.

    Parameters:
    - src_file (str): Path to the source raster file.
    - dst_file (str): Path to the destination HDF file.
    - dst_crs (str): Destination coordinate reference system (CRS) in EPSG format.
    - resolution (int): Resolution of the destination raster in meters (default: 30).
    - chunk_size (int): Chunk size for storing the data in the HDF file (default: 512).

    Returns:
    None
    """

    # Calculate the transform and shape of the destination raster
    with rasterio.open(src_file) as src:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=resolution)

        # Create the destination HDF with the same shape and dtype as the source raster
        with h5py.File(dst_file, 'w') as dst:
            # Write the metadata to the HDF
            dst.create_dataset('Affine', data=dst_transform.to_gdal(), dtype='float64')
            # Write the data to the HDF
            dst.create_dataset('array',
                               shape=(src.count, dst_height, dst_width),
                               dtype=src.dtypes[0],
                               chunks=(src.count, chunk_size, chunk_size),
                               compression='gzip',
                               fillvalue=0)


def reproject_raster(src_file, dst_file, dst_crs):
    """
    Reprojects a raster from the source CRS to the destination CRS.

    Args:
        src_file (str): The path to the source raster file.
        dst_file (str): The path to the destination HDF file.
        dst_crs (str): The destination CRS in WKT format.

    Returns:
        None
    """

    # Open the source raster
    with rasterio.open(src_file) as src:

        blocks = list(src.block_windows())

        # Read the source raster block by block
        for _, window in tqdm(blocks,total=len(blocks)):
            src_data = src.read(window=window)
            src_crs = src.crs
            src_transform = src.window_transform(window)
            src_width = window.width
            src_height = window.height
            src_bounds = src.window_bounds(window)

            # Calculate the transform and shape of the destination raster
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, dst_crs, src_width, src_height, *src_bounds, resolution=30)

            # Reproject the source raster to the destination raster   
            dst_data = np.zeros((src.count, dst_height, dst_width),dtype=src_data.dtype)

            dst_data_proj,_ = reproject(
                source=src_data,
                destination=dst_data,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)

            # Write the data to the destination HDF
            row_accumulator = 0
            col_accumulator = 0
            with h5py.File(dst_file,'r+') as dst:
                # Write the data to the HDF
                row_slice = slice(row_accumulator, row_accumulator + dst_height)
                col_slice = slice(col_accumulator, col_accumulator + dst_width)
                dst['array'][slice(None),row_slice,col_slice] = dst_data_proj

                # Update the row and column accumulators
                row_accumulator += dst_height
                col_accumulator += dst_width


def mosaic_tif_2_vrt(raster_files, vrt_path):
    
    if os.path.isfile(vrt_path):
        print('Skip existing: ', vrt_path)
        
    else:
        # Create a VRT
        vrt = gdal.BuildVRT(vrt_path, raster_files)
        vrt = None  # This will close and save the VRT


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
        
        
        
def reproject_raster(src_file, ref_file, dst_file):
    """
    Reprojects a raster file to match the coordinate reference system (CRS) of a reference file.

    Args:
        ref_file (str): The path to the reference file with the desired CRS.
        dst_file (str): The path to the output file where the reprojected raster will be saved.

    Returns:
        None
    """

    if os.path.exists(dst_file):
        print(f'Skip existing: {dst_file}')
    else:
    
        # Get the CRS of the reference file
        dst_crs = rasterio.open(ref_file).crs
        block_size = rasterio.open(ref_file).block_shapes[0][0]

        # Reproject the raster file
        with rasterio.open(src_file) as src:
            # Create a WarpedVRT instance
            with WarpedVRT(src, crs=dst_crs, resampling=Resampling.nearest) as vrt:
                profile = vrt.profile
                profile.update(driver='GTiff',
                            blockxsize=block_size, 
                            blockysize=block_size,
                            compression='lzw')
                # Write the reprojected raster to disk
                with rasterio.open(dst_file, 'w', **profile) as dst:
                    # Loop through the blocks
                    windows = list(vrt.block_windows())
                    for ji, window in tqdm(windows,total=len(windows)):
                        vrt_array = vrt.read(window=window)
                        # Write each band separately
                        for band_index in range(1, vrt.count + 1):
                            dst.write(vrt_array, window=window)