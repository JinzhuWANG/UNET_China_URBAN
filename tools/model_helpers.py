
# function to evaluate the model
import os
import numpy as np
import pandas as pd
import h5py
from glob import glob
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from STEP_00_parameters import BLOCK_SIZE, PAD_SIZE


# Set up working directory
if __name__ == '__main__':
    os.chdir('..')


# set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def eval_model(model, criterion,loader):
    """
    Function to evaluate the model.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        loader (torch.utils.data.DataLoader): The data loader for evaluation.

    Returns:
        list: A list of losses computed during evaluation.
    """

    # set to eval mode
    model.eval()

    # compute losses
    losses = []
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader)):

            # get data, then send them to GPU
            x = data[0].float().to(device)
            y = data[1].squeeze(1).float().to(device)

            # train the model
            score = model(x).squeeze(1)

            # compute loss
            loss = criterion(score, y)
            losses.append(loss.detach().cpu().item())

    # change model back to training mode    
    model.train()

    return losses



# load the trained modle
def load_saved_model(model):
    """
    Loads the saved model and returns the start epoch and best loss.

    Args:
        model: The model to load the saved weights into.

    Returns:
        start_epoch (int): The starting epoch of the loaded model.
        best_loss (float): The best loss achieved during training.
    """
    
    longest_trained_model = sorted(glob(f'data/Saved_models/Progress_model*'))
    if len(longest_trained_model) > 0:

        # load historical training models
        longest_trained_model = sorted(longest_trained_model)[-1]
        model.load_state_dict(torch.load(longest_trained_model,map_location=torch.device(device)))
        start_epoch = int(longest_trained_model[-7:-4])

        metrics_df = pd.read_csv('data/Metrics_csv/metrics.csv',header=None)
        best_loss = metrics_df[metrics_df[1]=='eval'][2].min()
        
        # report the start epoch
        print(f'Loaded model from {longest_trained_model}')
        print(f'Epoch ==> {start_epoch}')
        print(f'Best loss ==> {best_loss:.5f}')
        
    else:
        start_epoch = 0
        best_loss = 1e9
        # empty the recored folders
        files = [i for i in glob('data/Saved_models/Saved_models/*')]
        files.extend(glob('data/Metrics_csv/*'))
        files.extend(glob('data/Saved_models/Train_model_pred_imgs/*.jpeg'))
        for i in files:
            os.remove(i)
            print(f'{i} has been deleted!')
            
    return start_epoch, best_loss


def pred_one_img(model, dataset, idx, epoch, img_path='data/Train_model_pred_imgs'):
    """
    Predicts and saves the predicted and true images for a single image in the dataset.

    Args:
        model (torch.nn.Module): The trained model for image prediction.
        dataset (torch.utils.data.Dataset): The dataset containing the images.
        idx (int): The index of the image in the dataset to predict.
        epoch (int): The current epoch number.
        img_path (str, optional): The path to save the predicted images. Defaults to 'data/Train_model_pred_imgs'.

    Returns:
        None
    """
    x_arry, y_arry = dataset[idx]
    x_arry = x_arry.unsqueeze(0).float().to(device)
    y_arry = y_arry.squeeze(0).detach().numpy()

    # passing the img to model
    pred_img = model(x_arry).cpu().detach().numpy()[0, 0, :, :]

    # save true-pred img to disk
    concat_img = np.hstack([pred_img, y_arry])
    save_image(torch.tensor(concat_img), f"{img_path}/train_pred_img_{epoch:03}.jpeg")


def pad_chunk(hdf:str, chunks:list, idx:int):
    """
    Pad a chunk of an HDF file array to the size of (BLOCK_SIZE + PAD_SIZE).
    !!! If the chunk is on the edge of the array, then the padding it with 0 to meet the size requiremet.

    Parameters:
    - hdf (str): The path to the HDF file.
    - chunks (list): A list of chunks.
    - idx (int): The index of the chunk to pad.

    Returns:
    - arr (ndarray): The padded chunk array.
    """

    # Read the HDF file
    src = h5py.File(hdf, 'r')
    # Get the array
    src_array = src['array']

    # Get the chunk
    chunk = chunks[idx]
           
    # Ensure the chunk to be (BLOCK_SIZE, BLOCK_SIZE)
    chunk = (slice(None),
                slice(chunk[1].start, chunk[1].start + BLOCK_SIZE),
                 slice(chunk[2].start, chunk[2].start + BLOCK_SIZE))
    
    #############################################################################
    #                   the chunk is totally outside the array                  #
    #############################################################################
    
    if chunk[1].stop < 0 \
        or chunk[1].start > src_array.shape[1] \
        or chunk[2].stop < 0 \
        or chunk[2].start > src_array.shape[2]:
        return np.zeros((src_array.shape[0], BLOCK_SIZE + 2 * PAD_SIZE, BLOCK_SIZE + 2 * PAD_SIZE))

    
    #############################################################################
    #                 the chunk is partially touces the array                   #
    #############################################################################
    
    # renew the chunk slices so that it is PAD_SIZE bigger than original
    chunk_pad = [[0,0],
                 [chunk[1].start - PAD_SIZE, chunk[1].stop + PAD_SIZE],
                 [chunk[2].start - PAD_SIZE, chunk[2].stop + PAD_SIZE]]

    # innitialize the padding size to be 0
    pad_left = pad_right = pad_top = pad_bottom = 0

    # if the left edge of the chunk_pad is smaller than 0, 
    # then pad the arr_left with 0 of size -chunk_pad[1][0]
    if chunk_pad[1][0] < 0:
        pad_left = -chunk_pad[1][0]
        chunk_pad[1][0] = 0

    # if the right edge of the chunk_pad is bigger than the src_shape (src_array.shape[1]), 
    # then pad the arr_right with 0 of size chunk_pad[1][1] - src_array.shape[1]
    if chunk_pad[1][1] > src_array.shape[1]:
        pad_right = chunk_pad[1][1] - src_array.shape[1]
        chunk_pad[1][1] = src_array.shape[1]

    # if the top edge of the chunk_pad is smaller than 0,
    # then pad the arr_top with 0 of size -chunk_pad[2][0]
    if chunk_pad[2][0] < 0:
        pad_top = -chunk_pad[2][0]
        chunk_pad[2][0] = 0

    # if the bottom edge of the chunk_pad is bigger than the src_shape (src_array.shape[2]),
    # then pad the arr_bottom with 0 of size chunk_pad[2][1] - src_array.shape[2]
    if chunk_pad[2][1] > src_array.shape[2]:
        pad_bottom = chunk_pad[2][1] - src_array.shape[2]
        chunk_pad[2][1] = src_array.shape[2]

    # Get the array with valid values
    arr = src_array[:,
                    chunk_pad[1][0]:chunk_pad[1][1],
                    chunk_pad[2][0]:chunk_pad[2][1]]

    # If any of the padding size is not 0, then pad the array 
    # to meet the paded chunk size of (BLOCK_SIZE + 2 * PAD_SIZE)
    if any([pad_left, pad_right, pad_top, pad_bottom]):
        arr = np.pad(arr, ((0, 0),
                           (pad_left, pad_right),
                           (pad_top, pad_bottom)),
                           mode='constant',
                           constant_values=0)
        
    return arr


