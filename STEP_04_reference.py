
from glob import glob
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

from STEP_00_parameters import IN_CHANNLE
from tools.UNET import UNET
from tools.model_helpers import pad_chunk


# set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################################################
#                       Load model                         #
############################################################
model = UNET()

# change the input_layer to match the input data
model.downs[0] = nn.Sequential(
                nn.Conv2d(IN_CHANNLE, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU())

model.to(device)


# load the best model weights
best_models = glob('data/Saved_models/Best*')
model = torch.load(best_models[-1])


############################################################
#                         Get data                         #
############################################################

# Get all HDF files
HDFs = glob('data/raster/*.hdf')

hdf = h5py.File(HDFs[-1],'r')
hdf_chunks = list(hdf['array'].iter_chunks())


from rasterio import Affine

# Get the Affine matrix
trans = Affine(*hdf['transform'][:])

# Get the col/row index of the chunk
chunk = hdf_chunks[0]

top_left_col_row = (chunk[2].start, chunk[1].start)

top_left_xy = ~trans * top_left_col_row










arr = pad_chunk(HDFs[0], hdf_chunks, 0)

class X_y_dataset(Dataset):
    def __init__(self,HDFs,ref_raster):
        self.HDFs = HDFs
        self.chunks = list(h5py.File(ref_raster,'r')['array'].iter_chunks())
          
    def __getitem__(self, index):
        for hdf in self.HDFs:
            hdf_chunks = chunk_ref2hdf(hdf, self.chunks)
            arr = pad_chunk(hdf, hdf_chunks, index)
            yield arr










