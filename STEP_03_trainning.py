import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from STEP_00_parameters import IN_CHANNLE, NUM_EPOCH, PRED_IMG_IDX
from STEP_02_dataloader import train_dataloader, val_dataloader,\
                               all_dataset, toy_dataloader

from tools.UNET import UNET
from tools.model_helpers import eval_model, load_saved_model, pred_one_img


############################################################
#              Set up working parameters                   #
############################################################


# set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'



# Check if the model is trained on toy dataset
toy_train = False # True: train on toy dataset, False: train on full dataset

if toy_train:
    train_dataloader = toy_dataloader
    val_dataloader = toy_dataloader
    all_dataset = toy_dataloader.dataset
    PRED_IMG_IDX = len(all_dataset) - 1




############################################################
#                     Defining model                       #
############################################################

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


# Define loss function
criterion = torch.nn.BCELoss()

# Difine optimizer
optimizer = torch.optim.Adam(model.parameters())


############################################################
#                       Train model                        #
############################################################

# Check if there is a saved model, if yes, load the longest trained model
start_epoch, best_loss = load_saved_model(model)

# train the model
for epoch in range(start_epoch,NUM_EPOCH+1):
    
    train_losses = []
    for data in tqdm(train_dataloader,total=len(train_dataloader)):
        # model(data)
        X_data = data[0].float().to(device)
        y = data[1].squeeze(1).float().to(device)

        # prediction
        y_hat = model(X_data).squeeze(1)

        # compute loss
        loss = criterion(y_hat,y)

        # automatically update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # record the tranin loss
        train_losses.append(loss.detach().cpu().item())
    
    # compute the avg. train loss
    train_losses_mean = np.array(train_losses).mean()

    # compute the avg. test loss
    test_losses = eval_model(model, criterion,val_dataloader)
    test_losses_mean = np.array(test_losses).mean()

    # write metric to disk
    with open('data/Metrics_csv/metrics.csv', 'a') as f:
        f.write(f'{epoch},train,{train_losses_mean}\n')
        f.write(f'{epoch},eval,{test_losses_mean}\n')

    # pred a picture and save it to disk
    pred_one_img(model, all_dataset, PRED_IMG_IDX, epoch)

    # save models to disk
    model_path = 'data/Saved_models'

    # save best models
    if test_losses_mean < best_loss:
        # update best_loss
        best_loss = test_losses_mean
        # save model to disk
        torch.save(model.state_dict(), f'{model_path}/Best_model_{epoch:03}.tar')

    # save model for every 5 eopch
    if (epoch)%5 == 0:
        # save model to disk
        torch.save(model.state_dict(), f'{model_path}/Progress_model_{epoch:03}.tar')


    # report the training process
    print(f"Epoch {epoch:03}: ==> Train:{train_losses_mean:.5f} ==> Eval:{test_losses_mean:.5f}\n")