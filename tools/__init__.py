from glob import glob
import os
import pandas as pd
from rasterio import Affine
from tqdm.auto import tqdm

import h5py
import numpy as np
import geopandas as gpd

import pyproj
from osgeo import gdal
from shapely.geometry import box

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from tools.helper_func import tif2hdf



# Set up working directory
if __name__ == '__main__':
    os.chdir('../')


def get_sample_pts(ref_tif, ROI_box):
    """
    1) Project the shapefile to the same CRS as the reference raster, 
    2) Get the centroid of each polygon as the top-left anchor of tile samples.
    3) Remove points that are outside the bounds of the reference raster.

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




def mosaic_tif_2_vrt(raster_files, vrt_path):
    
    if os.path.isfile(vrt_path):
        print('Skip existing: ', vrt_path)
        
    else:
        # Create a VRT
        vrt = gdal.BuildVRT(vrt_path, raster_files)
        vrt = None  # This will close and save the VRT


def reproject_raster(src_file, ref_file, dst_file):
    """
    Reprojects a raster file to match the coordinate reference system (CRS) of a reference file.

    Args:
        src_file (str): The path to the source raster file.
        ref_file (str): The path to the reference raster file.
        dst_file (str): The path to save the reprojected raster file.

    Returns:
        None
    """

    if os.path.exists(dst_file):
        print(f'Skip existing: ',  dst_file)
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
                    for _, window in tqdm(windows, total=len(windows)):
                        vrt_array = vrt.read(window=window)
                        # Write each band separately
                        dst.write(vrt_array, window=window)


def write_slices_2_HDF(hdf_file:str,sample_pts:pd.DataFrame):
    """
    Write slices to an HDF file based on sample points.

    Parameters:
    hdf_file (str): The path to the HDF file.
    sample_pts (pd.DataFrame): The DataFrame containing sample points.

    Returns:
    None
    """

    # Read the raster (HDF) file, get transform and ndarray
    with h5py.File(hdf_file, 'r+') as src:

        trans = Affine(*src['transform'])
        array = src['array']
        chunk_size = array.chunks[-1]


        # Loop through each sample point, and compute 
        # the slices for each point
        row_slices = []
        col_slices = []
        for i, row in sample_pts.iterrows():
            # Get the sample point coordinate
            x, y = row['centroid'].x, row['centroid'].y

            # Get the slice for the sample point
            col,row =  ~trans * (x, y)
            col,row = int(col), int(row)

            row_slice = [row, row + chunk_size]
            col_slice = [col, col + chunk_size]

            # Append the slices to the list
            row_slices.append(row_slice)
            col_slices.append(col_slice)

        # Write the slices to the HDF file
        if 'row_slices' in src.keys():
            del src['row_slices']
            del src['col_slices']
            
        src.create_dataset('row_slices', data=np.array(row_slices))
        src.create_dataset('col_slices', data=np.array(col_slices))
        
        # Report the number of slices
        print(f'Number of slices added: {len(row_slices)} ==> {hdf_file}')