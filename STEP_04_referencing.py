
from glob import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm

from STEP_00_parameters import BATCH_SIZE, IN_CHANNLE, NUM_WORKERS, YR_TRAIN_FROM, YR_TRAIN_TO
from tools.UNET import UNET
from tools.helper_func import chunk_transform
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

class all_data_chunk_array(Dataset):
    
    def __init__(self,HDFs, yr_from, yr_to):
        super().__init__()
        self.yr_from = yr_from
        self.yr_to = yr_to
        self.HDFs = HDFs
        
        # get the HDF file paths
        self.LUCC_from = [i for i in self.HDFs if str(self.yr_from) in i][0]
        self.terrain = [i for i in self.HDFs if 'terrain' in i][0]
        
        # use the LUCC_from as geo_ref to get chunks for each hdf
        self.LUCC_from_chunks = list(h5py.File(self.LUCC_from, 'r')['array'].iter_chunks())
        self.terrain_chunks = chunk_transform(self.LUCC_from, self.terrain)
              
    def __getitem__(self, index):
        
        # pad the chunk
        lucc_arr = pad_chunk(self.LUCC_from,self.LUCC_from_chunks,index)
        terrain_arr = pad_chunk(self.terrain,self.terrain_chunks,index)
        
        out_arr = np.concatenate((lucc_arr,terrain_arr),axis=0)
        
        return torch.FloatTensor(out_arr)
    
    def __len__(self):
        return len(self.LUCC_from_chunks)
        
        
        
# create the dataset
HDFs = glob('data/raster/*.hdf') 
  
all_arries = all_data_chunk_array(HDFs, YR_TRAIN_FROM, YR_TRAIN_TO)
all_arries_loader = DataLoader(all_arries, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False, num_workers=NUM_WORKERS)


# for idx,arr in tqdm(enumerate(all_arries_loader),total=len(all_arries_loader)):
#     print(arr.shape)

        








