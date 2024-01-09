import h5py
from glob import glob
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader,Dataset

# from torchvision.utils import save_image
from sklearn.model_selection import train_test_split

from STEP_00_parameters import BATCH_SIZE, IMPERVIOUS_VAL, TOY_SIZE, \
                               YR_TRAIN_FROM, YR_TRAIN_TO

    
    
#####################################################
#             Create the dataset function           #
#####################################################


def get_arrary_from_hdf(hdf_path:str, indexes, idx:int, toy_size=None):
    """
    Retrieves a subset of an array from an HDF file based on given indexes.

    Parameters:
    - hdf_path (str): The path to the HDF file.
    - indexes: The indexes used to subset the slices.
    - idx (int): The index of the subseted slices.

    Returns:
    - arrary: The subset of the array from the HDF file.
    """
    
    # Read HDF file
    hdf = h5py.File(hdf_path,'r')
    
    # Subset the slices using the indexes
    row_slices = hdf['row_slices'][:][indexes]
    col_slices = hdf['col_slices'][:][indexes]
    
    # Get the slice
    row_slice = row_slices[idx]
    col_slice = col_slices[idx]
    
    # Get the array
    arrary = hdf['array'][slice(None),
                          slice(*row_slice),
                          slice(*col_slice)]
    
    # Shrink the size if toy_size is given
    if toy_size:
        arrary = arrary[:,
                        0:toy_size,
                        0:toy_size]
    
    # Close HDF file
    hdf.close()
    return arrary



# dataset class
class X_y_dataset(Dataset):
    def __init__(self,HDFs,indexes,yr_from,yr_to, toy_size=None):
        
        self.indexes = indexes
        self.yr_from = yr_from
        self.yr_to = yr_to
        self.HDFs = HDFs
        self.toy_size = toy_size
        
    def __getitem__(self,index):
        
        # get the HDF file paths
        LUCC_from = [i for i in self.HDFs if str(self.yr_from) in i][0]
        LUCC_to = [i for i in self.HDFs if str(self.yr_to) in i][0]
        terrain = [i for i in self.HDFs if 'terrain' in i][0]
        
        # slice array from HDF file
        LUCC_from_arr = get_arrary_from_hdf(LUCC_from,self.indexes,index,self.toy_size)
        LUCC_to_arr = get_arrary_from_hdf(LUCC_to,self.indexes,index,self.toy_size)
        terrain_arr = get_arrary_from_hdf(terrain,self.indexes,index,self.toy_size)
        
        # Get the impervious surface array (value = 8)
        impervious_arr_from = np.where(LUCC_from_arr==IMPERVIOUS_VAL,1,0)
        impervious_arr_to = np.where(LUCC_to_arr==IMPERVIOUS_VAL,1,0)
        
        # stack the arrays
        X = np.vstack([impervious_arr_from,terrain_arr])
        
        # Return the X and y tensors
        return torch.FloatTensor(X),torch.FloatTensor(impervious_arr_to)
    
    def __len__(self):
        return len(self.indexes)
        
        
#####################################################
#     Randomly splice sample (train_8k:val_2k)      #
#####################################################

# Get sample points
sample_pts = pd.read_csv('data/sample_pts.csv')

# Split the sample indexes into train and validation sets
train_idx, val_idx = train_test_split(range(len(sample_pts)),
                                      test_size=0.2, 
                                      random_state=42)

# Get all HDF files
HDFs = glob('data/raster/*.hdf')

# Get the train and validation datasets, dataloaders
all_dataset   = X_y_dataset(HDFs,range(len(sample_pts)),YR_TRAIN_FROM,YR_TRAIN_TO)
toy_dataset   = X_y_dataset(HDFs,range(100),YR_TRAIN_FROM,YR_TRAIN_TO, toy_size=TOY_SIZE)

train_dataset = X_y_dataset(HDFs,train_idx,YR_TRAIN_FROM,YR_TRAIN_TO)
val_dataset   = X_y_dataset(HDFs,val_idx,YR_TRAIN_FROM,YR_TRAIN_TO)

train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
val_dataloader   = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True)
toy_dataloader   = DataLoader(toy_dataset,batch_size=BATCH_SIZE,shuffle=True)  # for testing



