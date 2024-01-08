import torch
import torch.nn as nn

from STEP_00_parameters import IN_CHANNLE
from tools.UNET import UNET

# set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the model
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