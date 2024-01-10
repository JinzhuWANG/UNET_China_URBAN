from glob import glob

from tools import mosaic_tif_2_vrt, get_sample_pts, \
                  reproject_raster, write_slices_2_HDF
                  
from tools.helper_func import tif2hdf



#####################################################
#                  Convert LUCC to HDF              #
#####################################################
 

# Get LUCC GeoTIFF files 
LUCC_files = glob('data/raster/CLCD_*.tif')

# Convert LUCC GEOTIFF files to HDF
for file in LUCC_files:
        tif2hdf(file)

        
#####################################################
#           Mosaic Terrain tifs to VRT              #
#####################################################

# List of raster files to mosaic
raster_files = glob('data/raster/China_DEM_SLOPE/*.tif') 
# Path for the output VRT file
terrain_vrt_path = 'data/raster/terrain.vrt'

mosaic_tif_2_vrt(raster_files, terrain_vrt_path)



#####################################################
#         Reproject Terrain VRT to LUCC.crs         #
#####################################################

# Get input/output files 
ref_file = LUCC_files[0]
dst_file = 'data/raster/terrain_proj2ref.tif'

# Reproject the raster file
reproject_raster(terrain_vrt_path, ref_file, dst_file)

# Convert the reprojected raster to GeoTIFF
tif2hdf(dst_file)


#####################################################
#      Create slice index from sample points        #
#####################################################

# Use ROI_box to restrict the sample points generation
ROI_box = 'data/vector/China_ROI_rect_sub_10k.shp'

# Get the sample points, indicating the top-left 
# coordinate of each tile
sample_pts = get_sample_pts(LUCC_files[0], ROI_box)

# Save the sample points to a CSV file
sample_pts.to_csv('data/sample_pts.csv', index=False)

# Write the sample slices to HDFs
HDFs = glob('data/raster/*.hdf')
for hdf_file in HDFs:
    write_slices_2_HDF(hdf_file, sample_pts)




    
