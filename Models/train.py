import torch
from tqdm import tqdm
import torch.nn as nn
import wandb
from torchmetrics import JaccardIndex
from utils import(
    evaluate
)
import rasterio
import numpy as np
import os
import glob
import torchvision
from torcheval.metrics.functional import mean_squared_error


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y):
        dice_score = (2. * (y_pred * y).sum() + 1.) / ((y_pred + y).sum() + 1.)

        return 1. - dice_score


def train(model, train_loader, val_loader, loss_function, optimizer, scaler, num_epochs, device, FILENAME, SAVE_MODEL, LOCAL, PLOT, three_channels, PRETRAIN=False):
    jaccard = JaccardIndex(task='binary').to(device)
    for epoch in range(num_epochs):
        tot_loss = 0
        tot_iou = 0
        tot_dice = 0
        tot_mse = 0
        model.train()
        with tqdm(train_loader, unit="batch") as batch:
            for x, y in batch:
                batch.set_description(f"Epoch {epoch}")
                x = x.to(device)
                y = y.float().unsqueeze(1).to(device)

                with torch.cuda.amp.autocast():
                    y_pred = model(x)
                    if PRETRAIN:
                        loss = loss_function(y_pred, x)
                    else:
                        loss = loss_function(y_pred, y)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch.set_postfix(loss=loss.item())
                tot_loss += loss.item()
                if PRETRAIN:
                    tot_mse += mean_squared_error(y_pred.flatten(), x.flatten())
                else:
                    tot_iou += jaccard(y_pred, y)
                    tot_dice += (2. * (y_pred * y).sum()) / ((y_pred + y).sum() + 1e-8)

        # evaluate model
        avg_loss = tot_loss / len(train_loader)
        if PRETRAIN:
            avg_mse = tot_mse / len (train_loader)
            train_wandb_log = {'train_loss': avg_loss, 'train_mse': avg_mse}
        else:
            avg_iou = tot_iou / len(train_loader)
            avg_dice = tot_dice / len(train_loader)
            train_wandb_log = {'train_loss': avg_loss, 'train_iou': avg_iou, 'train_dice': avg_dice}

        val_wandb_log = evaluate(model, val_loader, loss_function, device, three_channels, PLOT, PRETRAIN)
        wandb_log = {**train_wandb_log, **val_wandb_log}
        if not LOCAL:
            wandb.log(wandb_log)

    # save model
    if SAVE_MODEL:
        if PRETRAIN:
            checkpoint = {
                'state_dict': model.encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            filename = FILENAME + str(num_epochs) + '_epochs'
            torch.save(checkpoint, filename)
        else:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            filename = FILENAME + str(num_epochs) + '_epochs'
            torch.save(checkpoint, filename)

    return model


def final_evaluation(model, test_loader, loss_function, device, mask_folder, LOCAL, three_channels):
    log = evaluate(model, test_loader, loss_function, device, three_channels, False)
    print('----- Model performance -----')
    print('IoU = ' + str(log['val_iou'].item()))
    print('Dice = ' + str(log['val_dice'].item()))

    year = '2020'
    month = '08'
    annotation = reconstructed_mask(mask_folder, year, month)
    prediction = reconstructed_prediction(model, test_loader, device, year, month)
    image_array = [torch.Tensor(np.transpose(annotation, (2, 0, 1))).to(device), prediction.permute(2, 0, 1)]

    image_array = torchvision.utils.make_grid(image_array, nrow=1)
    image = wandb.Image(image_array, caption="ground truth, prediction")
    wandb_log = {"examples": image}
    if not LOCAL:
        wandb.log(wandb_log)


def reconstructed_prediction(model, loader, device, YEAR, MONTH):
    model.eval()
    for i, (x, y) in enumerate(loader):
        file = loader.dataset.target[i * loader.batch_size]
        year = ((file.split('/')[10]).split('_')[3]).split('-')[0]
        month = ((file.split('/')[10]).split('_')[3]).split('-')[1]
        if year == YEAR and MONTH == month:
            with torch.no_grad():
                x = x.to(device)

                y_pred = model(x)
                y_pred = (y_pred > 0.5).float()
                y_pred = y_pred.to(device)

                i = 0
                j = 0

                print(y_pred.shape)
                for k in range(y_pred.shape[0]):
                    mask = y_pred[k]
                    print(mask)
                    mask = mask.repeat(3, 1, 1)
                    mask = mask.permute(1, 2, 0)
                    if i == 0:
                        row = mask
                    else:
                        row = torch.cat((row, mask), 0)

                    i += 1
                    if i == 7:
                        if j == 0:
                            image = row
                        else:
                            image = torch.cat((image, row), 1)
                        j += 1
                        i = 0
    return image


def reconstructed_mask(folder, YEAR, MONTH):
    i = 0
    j = 0

    filelist = glob.glob(os.path.join(folder, '*.tif'))

    for file in sorted(filelist):
        year = ((file.split('/')[10]).split('_')[3]).split('-')[0]
        month = ((file.split('/')[10]).split('_')[3]).split('-')[1]
        if year == YEAR and month == MONTH:
            dataset = rasterio.open(file)

            mask = dataset.read(1)
            mask = mask // 255
            mask = np.array([mask, mask, mask])
            mask = np.transpose(mask, (1, 2, 0))
            if i == 0:
                row = mask
            else:
                row = np.concatenate((row, mask), 0)

            i += 1
            if i == 7:
                if j == 0:
                    image = row
                else:
                    image = np.concatenate((image, row), 1)
                j += 1
                i = 0

    return image
