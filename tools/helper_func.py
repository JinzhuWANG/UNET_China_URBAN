
import os
import numpy as np
import rasterio
from rasterio import Affine
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
            
            
            
            
def chunk_transform(hdf_1:str, hdf_2:str):
    """
    Transforms the chunks of hdf_1 to their equivalent chunks in hdf_2 based on the resolution and transform attributes.

    Parameters:
    hdf_1 (str): Path to the first HDF file.
    hdf_2 (str): Path to the second HDF file.

    Returns:
    list: List of transformed chunks in hdf_2.
    """

    
    # read the hdf datasets
    hdf_dst_1 = h5py.File(hdf_1,'r')
    hdf_dst_2 = h5py.File(hdf_2,'r')
    
    # read the transform attribute
    transform_1 = list(hdf_dst_1['transform'][:])
    transform_2 = list(hdf_dst_2['transform'][:])
    
    # get the resolutions
    res_1 = transform_1[0]
    res_2 = transform_2[0]
    
    if res_1 != res_2:
        raise ValueError('The resolutions of the two HDF files are different')
    
    # calculate the col/row deltas
    delta_col = int((transform_2[2] - transform_1[2]) / res_1)
    delta_row = int((transform_2[5] - transform_1[5]) / res_1)
    
    # get the chunks from hdf_1
    hdf_1_chunks = list(hdf_dst_1['array'].iter_chunks())
    
    # transform the chunk of hdf_1 to its equivalent chunk in hdf_2
    hdf_2_chunks_transformed = [(
        slice(None),
        slice(chunk[1].start + delta_row, chunk[1].stop + delta_row),
        slice(chunk[2].start - delta_col, chunk[2].stop - delta_col)
    ) for chunk in hdf_1_chunks]
    
    return hdf_2_chunks_transformed


# def rowcol_transform(hdf_1, hdf_2, row_col_1):
#     """
#     Transforms column and row indices from one raster to another raster's spatial coordinates.

#     Parameters:
#         hdf_1 (str): Path to the first HDF file.
#         hdf_2 (str): Path to the second HDF file.
#         col_row_1 (tuple): Column and row indices in the first raster.

#     Returns:
#         tuple: Transformed column and row indices in the second raster.
#     """

#     # Get the transforms
#     with h5py.File(hdf_1) as src:
#         transform_1 = list(src['transform'][:])
#         affine_1 = Affine(*transform_1)
#         # Convert row and column indices from raster 1 to spatial coordinates
#         x, y = affine_1 * (row_col_1[1], row_col_1[0])

#     with h5py.File(hdf_2) as src:
#         transform_2 = list(src['transform'][:])
#         affine_2 = Affine(*transform_2)
#         row_col_2 = ~affine_2 * (x, y)

#     return int(row_col_2[0]), int(row_col_2[1])


# def chunk_transform(hdf_1, hdf_2):
#     """
#     Transforms a chunk from one HDF file to its equivalent in another HDF file.

#     Parameters:
#     - hdf_1 (str): Path to the first HDF file.
#     - hdf_2 (str): Path to the second HDF file.

#     Returns:
#     - chunk_transformed_all (list): List of transformed chunks.

#     """
#     # get chunks from hdf_1
#     hdf_1_chunks = list(h5py.File(hdf_1, 'r')['array'].iter_chunks())

#     chunk_transformed_all = []
#     # transform the chunk to its equivalent in hdf_2
#     for chunk in hdf_1_chunks:

#         # get the top left and bottom right rowcol of the chunk
#         rowcol_top_left = (chunk[1].start, chunk[2].start)
#         rowcol_bottom_right = (chunk[1].stop, chunk[2].stop)

#         # transform the rowcol to the other HDF
#         rowcol_top_left_transformed = rowcol_transform(hdf_1, hdf_2, rowcol_top_left)
#         rowcol_bottom_right_transformed = rowcol_transform(hdf_1, hdf_2, rowcol_bottom_right)

#         # create the transformed chunk
#         chunk_transformed = (chunk[0],
#                              slice(rowcol_top_left_transformed[0], rowcol_bottom_right_transformed[0]),
#                              slice(rowcol_top_left_transformed[1], rowcol_bottom_right_transformed[1]))

#         chunk_transformed_all.append(chunk_transformed)

#     return chunk_transformed_all


