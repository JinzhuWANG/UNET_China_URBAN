import os
from glob import glob
import rasterio

from tools import GEOTIF2HDF, create_empty_raster, remove_out_bounds_pts, reproject_raster


# Get LUCC GeoTIFF files 
LUCC_files = glob('data/raster/*.tif')

# Convert LUCC GEOTIFF files to HDF
for file in LUCC_files:
    # skip if the file is already converted
    if os.path.isfile(file.replace('.tif', '.hdf')):
        print('Skip existing: ', file)
        continue
    else:
        GEOTIF2HDF(file)
        



# # Built a VRT from the DEM tiles
# from osgeo import gdal

# # List of raster files to mosaic
# raster_files = glob('data/raster/China_DEM_SLOPE/*.tif') 

# # Path for the output VRT file
# vrt_path = 'data/raster/terrain.vrt'

# # Create a VRT
# vrt = gdal.BuildVRT(vrt_path, raster_files)
# vrt = None  # This will close and save the VRT




import rasterio
import pyproj

# Get the destination CRS 
ref_file = LUCC_files[0]
dst_crs = pyproj.CRS(rasterio.open(ref_file).crs)

# Reproject the VRT to the destination CRS
src_file = 'data/raster/terrain.vrt'
dst_file = 'data/raster/terrain_proj2ref.hdf'

create_empty_raster(src_file, dst_file, dst_crs, resolution=30, chunk_size=512)
reproject_raster(src_file, dst_file, dst_crs)

# Get the top-left coordinate of sample tiles
ROI_box = 'data/vector/China_ROI_rect_sub_10k.shp'
sample_top_left = remove_out_bounds_pts(LUCC_files[0], ROI_box)




