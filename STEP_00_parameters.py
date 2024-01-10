# Data info
IMPERVIOUS_VAL = 8

# Define the number of workers for dataloader
NUM_WORKERS = 12 

# The block size for each tile image
BLOCK_SIZE = 512
PAD_SIZE = 128

# Historical traning years
YR_TRAIN_FROM = 2006
YR_TRAIN_TO = 2014

# YR_TRAIN_FROM = 2014
# YR_TRAIN_TO = 2022

# Model info
BATCH_SIZE = 8
NUM_EPOCH = 400
IN_CHANNLE = 3   # 1) impervious surface, 2) DEM, 3) SLOPE


# Define the size for toy dataset
TOY_SIZE = 32

# The index of the image to be predicted/saved during training
PRED_IMG_IDX = 3206  # Determined by manual inspection of the dataset
