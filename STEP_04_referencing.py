
from glob import glob
import h5py
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm

from tools.UNET import UNET
from tools.helper_func import chunk_transform
from tools.model_helpers import pad_chunk

from STEP_00_parameters import BLOCK_SIZE, IMPERVIOUS_VAL, IN_CHANNLE, NUM_WORKERS, PAD_SIZE, \
                               YR_TRAIN_FROM, YR_TRAIN_TO

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

# load the best model weights
model_weight = sorted(glob('data/Saved_models/Best*'))[-1]
model_name = model_weight.split('/')[-1].split('.')[0]
model_weight = torch.load(model_weight)
model.load_state_dict(model_weight)

print(f"Use the model: {model_name}")

# send the model to device
model.to(device)
model.eval()


############################################################
#                      Create Dataset                      #
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
        
        chunk = self.LUCC_from_chunks[index]
        
        # Calculate window
        row_start = chunk[1].start
        row_stop = chunk[1].stop
        col_start = chunk[2].start
        col_stop = chunk[2].stop   

        window = dict(row_off=row_start,
                      col_off=col_start, 
                      width=col_stop - col_start, 
                      height=row_stop - row_start)
        
        # pad the chunk
        lucc_arr = pad_chunk(self.LUCC_from,self.LUCC_from_chunks,index)
        lucc_arr = np.where(lucc_arr==IMPERVIOUS_VAL,1,0)
        
        terrain_arr = pad_chunk(self.terrain,self.terrain_chunks,index)
        
        out_arr = np.concatenate((lucc_arr,terrain_arr),axis=0)
        
        return torch.FloatTensor(out_arr), window
    
    def __len__(self):
        return len(self.LUCC_from_chunks)
    
        
############################################################
#               Get DataLoader and predict                 #
############################################################       
        
# create the dataset
HDFs = glob('data/raster/*.hdf') 
  
all_arries = all_data_chunk_array(HDFs, YR_TRAIN_FROM, YR_TRAIN_TO)
all_arries_loader = DataLoader(all_arries, batch_size=1, pin_memory=True,num_workers=4)


# Use the LUCC TIF as template to create the predicted TIF
template = f'data/raster/CLCD_v01_{YR_TRAIN_FROM}_albert.tif'

# Get the meta data from the template
with rasterio.open(template) as src:
    meta = src.meta.copy()
    meta.update(dtype = np.int16, 
                compress =  'lzw',
                blockxsize = BLOCK_SIZE,
                blockysize = BLOCK_SIZE,
                BIGTIFF = "IF_SAFER")

    
# create the predicted TIF
with rasterio.open(f'data/predicted_{YR_TRAIN_FROM}_{YR_TRAIN_TO}_{model_name}.tif', 
                    'w', 
                    **meta) as dst:

    # predict and write to the predicted TIF
    model.eval()
    for idx,(arr,win) in tqdm(enumerate(all_arries_loader),
                              total=len(all_arries_loader)):
        
        # Get the window and array
        window = rasterio.windows.Window(**{k:v.numpy()[0] for (k,v) in win.items()})
        arr = arr.to(device)
        
        with torch.no_grad():
            pred = model(arr)
            # subset the array to (BATCH_SIZE, 1, BLOCK_SIZE, BLOCK_SIZE)
            pred = pred[0,
                        :,
                        PAD_SIZE: -PAD_SIZE,
                        PAD_SIZE: -PAD_SIZE]
            
            # convert to numpy array, rarange to (0, 10k), and change dtype to int16
            pred = pred.cpu().numpy() * 10000
            pred = pred.astype(np.int16)
            
            
            dst.write(pred,window=window) 
        

