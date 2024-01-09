
# function to evaluate the model
import os
import numpy as np
import pandas as pd
from glob import glob
import torch
from torchvision.utils import save_image
from tqdm import tqdm

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
    
    longest_trained_model = glob(f'data/Saved_models/Progress_model*')
    if len(longest_trained_model) > 0:

        # load historical training models
        longest_trained_model = sorted(longest_trained_model)[-1]
        model.load_state_dict(torch.load(longest_trained_model,map_location=torch.device(device)))
        start_epoch = int(longest_trained_model[-7:-4])

        metrics_df = pd.read_csv('data/Metrics_csv/metrics.csv',header=None)
        best_loss = metrics_df[metrics_df[1]=='eval'][2].min()
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