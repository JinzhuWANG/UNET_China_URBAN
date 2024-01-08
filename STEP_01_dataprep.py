import os
from glob import glob
import rasterio
from tqdm.auto import tqdm

from tools import GEOTIF2HDF, create_empty_raster, mosaic_tif_2_vrt, \
                  remove_out_bounds_pts, reproject_raster, vrt2tif



#####################################################
#                  Convert LUCC to HDF              #
#####################################################
 

# Get LUCC GeoTIFF files 
LUCC_files = glob('data/raster/*.tif')
# Convert LUCC GEOTIFF files to HDF
GEOTIF2HDF(LUCC_files)

        
#####################################################
#           Mosaic Terrain tifs to VRT              #
#####################################################

# List of raster files to mosaic
raster_files = glob('data/raster/China_DEM_SLOPE/*.tif') 
# Path for the output VRT file
terrain_vrt_path = 'data/raster/terrain.vrt'
terrain_tif_path = 'data/raster/terrain.tif'

mosaic_tif_2_vrt(raster_files, terrain_vrt_path)
vrt2tif(terrain_vrt_path, terrain_tif_path)



#####################################################
#         Reproject Terrain VRT to LUCC.crs         #
#####################################################


# Get input/output files 
ref_file = LUCC_files[0]
dst_file = 'data/raster/terrain_proj2ref.tif'

reproject_raster(terrain_tif_path, ref_file, dst_file)










# create_empty_raster(src_file, dst_file, dst_crs, resolution=30, chunk_size=512)
# reproject_raster(src_file, dst_file, dst_crs)

# # Get the top-left coordinate of sample tiles
# ROI_box = 'data/vector/China_ROI_rect_sub_10k.shp'
# sample_top_left = remove_out_bounds_pts(LUCC_files[0], ROI_box)

