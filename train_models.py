import json
import random
import os
import time
import matplotlib.pyplot as plt
import wandb
from dotenv import load_dotenv, dotenv_values
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import rasterio as rio
import itertools
from itertools import product
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms, datasets
from PIL import Image

from jaccard_similarity import calculate_intersection_over_union
import archs as arc
import SegNet as seg
import unet as un
import utils

class CFDDataset(Dataset):
    def __init__(self, dataset, images_dir, masks_dir,transform_img=None, transform_noise=None):
        self.dataset = dataset
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform_img = transform_img
        self.transform_noise = transform_noise

    def __getitem__(self, index):
        index_ = self.dataset.iloc[index]['id']

        # Get image and mask file paths for specified index
        image_path = self.images_dir + str(index_) + '-sar.tif'
        mask_path = self.masks_dir + str(index_) + '-mask.tif'

        # Read image
        img = rio.open(image_path).read()
        image_tensor = torch.from_numpy(img.astype(np.float32))

        # Read image
        mas = rio.open(mask_path).read()
        mask_tensor = torch.from_numpy(mas.astype(np.float32))

        if self.transform_img:
            img0 = self.transform_img(image_tensor)
            img = self.transform_noise(img0)
            mas = self.transform_img(mask_tensor)
            return img, mas

        return image_tensor, mask_tensor

    def __len__(self):
        return len(self.dataset)


def get_dataloaders(data, batch_size, num_workers, images_dir, masks_dir,train_transform_img=None, transform_msk=None):
    datasets = {
        'train' : CFDDataset(data[data.split == 'train'], images_dir, masks_dir, train_transform_img, transform_msk),
        'test' : CFDDataset(data[data.split == 'test'], images_dir, masks_dir)
    }

    dataloaders = {
        'train': DataLoader(
          datasets['train'],
          batch_size=batch_size,
          shuffle=True,
          num_workers=num_workers
        ),
        'test': DataLoader(
          datasets['test'],
          batch_size=batch_size,
          drop_last=False,
          num_workers=num_workers
        )
    }
    return dataloaders


class DiceLoss(nn.Module):
    def __init__(self, lambda_=1.):
        super(DiceLoss, self).__init__()
        self.lambda_ = lambda_

    def forward(self, y_pred, y_true):
        y_pred = y_pred[:, 0].view(-1)
        y_true = y_true[:, 0].view(-1)
        intersection = (y_pred * y_true).sum()
        dice_loss = (2. * intersection  + self.lambda_) / (
            y_pred.sum() + y_true.sum() + self.lambda_
        )
        return 1. - dice_loss

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def intersection_over_union(y_pred, y_true):

    smooth = 1e-6
    y_pred = y_pred[:, 0].view(-1) > 0.5
    y_true = y_true[:, 0].view(-1) > 0.5
    intersection = (y_pred & y_true).sum() + smooth
    union = (y_pred | y_true).sum() + smooth
    iou = intersection / union
    return iou


def train(model, dataloader, criterion, optimizer, device):
    model.train(True)
    losses = []
    ious = []
    criterion= criterion.to(device)

    for input, target in tqdm(dataloader, total=len(dataloader)):
        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            output = model(input)

            # UNET++ with deep supervision = True
            #loss = 0
            #for elem  in output:
            #    loss += criterion(elem, target)
            #loss /= len(output)

            #All other cases: UNET, SegNet and UNET++ with deep supervision = False
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            iou = intersection_over_union(output, target)
            losses.append(loss.cpu().detach().numpy())
            ious.append(iou.cpu().detach().numpy())

    train_loss = np.mean(losses)
    train_iou = np.mean(ious)

    metrics = {
        'train_loss': train_loss,
        'train_iou': train_iou,
    }

    return metrics


def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses = []
    ious = []

    for input, target in tqdm(dataloader, total=len(dataloader)):
        input = input.to(device)
        target = target.to(device)

        with torch.set_grad_enabled(False):
            output = model(input)

            # UNET++ with deep supervision = True
            #loss = 0
            #for elem in output:
            #    loss += criterion(elem, target)
            #    loss /= len(output)
            #iou = intersection_over_union(output[-1], target)

            #All other cases: UNET, SegNet and UNET++ with deep supervision = False
            loss = criterion(output, target)
            iou = intersection_over_union(output, target)

            losses.append(loss.cpu().detach().numpy())
            ious.append(iou.cpu().detach().numpy())

    val_loss = np.mean(losses)
    val_iou = np.mean(ious)

    metrics = {
        'val_loss': val_loss,
        'val_iou': val_iou,
    }

    return metrics


def save_model(model, model_dir, model_file):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_file)
    torch.save(model.state_dict(), model_path)
    print(f'Model successfully saved to {model_path}')

def load_model_func(model_file, device):
    loaded_model = UNET(in_channels=1, out_channels=1)
    loaded_model.to(device)
    loaded_model.load_state_dict(torch.load(model_file, map_location=device))
    loaded_model.eval()

    print('Model file {} successfully loaded.'.format(model_file))

    return loaded_model

def plot_single_sar_image(model, tiles_data, images_dir, ndwi_masks_dir, device):
    i = 120
    model.eval()
    index = tiles_data[tiles_data.split == 'test'].iloc[i]['id']
    image_path = images_dir + str(index) + '-sar.tif'
    image = rio.open(image_path).read()
    print(image.shape)


def evaluate_single_image(model, tiles_data, images_dir, ndwi_masks_dir, device):
    i = 100
    model.eval()
    index = tiles_data[tiles_data.split == 'test'].iloc[i]['id']
    image_path = images_dir + str(index) + '-sar.tif'

    sar_image = rio.open(image_path).read()
    sar_image = np.expand_dims(sar_image[0,:,:],axis=0)

    ndwi_image_path = ndwi_masks_dir + str(index) + '-mask.tif'
    ndwi_image = rio.open(ndwi_image_path).read()

    batch_sar_image = sar_image[None, :]
    batch_sar_image = torch.from_numpy(batch_sar_image.astype(np.float32)).to(device)
    pred_image = model(batch_sar_image).cpu().detach().numpy()

    pred_image = pred_image.squeeze()
    plt.imshow(pred_image)
    plt.show()
    plt.clf()

    iou = intersection_over_union(ndwi_image[0], pred_image[0])

    return sar_image, pred_image, ndwi_image, iou


def full_cycle(BS, LR, WD):
    n_epochs = int(os.getenv('EPOCHS'))
    learning_rate = float(LR)
    seed = int(os.getenv('RANDOM_SEED'))
    batch_size = int(BS)
    num_workers = int(os.getenv('NUM_WORKERS'))
    model_dir = os.getenv('MODELS_DIR')
    region = os.getenv('REGION_ASCII_NAME')
    date = os.getenv('START_DATE')
    polarization = os.getenv('SAR_POLARIZATION')
    orbit_pass = os.getenv('ORBIT_PASS')
    patch_size = int(os.getenv('PATCH_SIZE'))
    ndwi_input = os.getenv('NDWI_INPUT')

    wandb.login()

    config = {
        "learning_rate": learning_rate,
        "weight_decay":WD,
        "epochs": n_epochs,
        "patch_size": patch_size,
        "ndwi_input": ndwi_input,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "random_seed": seed,
        "region": region,
        "date": date,
        "polarization": polarization,
        "orbit_pass": orbit_pass,
        "regularization":'adamW',
        "Data Aug":'CORR(rot_3_5__05 + GN_05)'
    }

    wandb.init(project="UNET++_2020_DA", config=config)

    run_name = wandb.run.name
    utils.plant_random_seed(seed)

    images_dir = os.getenv('SAR_DIR') + '/'
    masks_dir = os.getenv('NDWI_MASK_DIR') + '/'
    tiles_data_file = os.getenv('TILES_FILE')

    # Check is GPU is enabled
    device = utils.get_device()

    tiles_data = pd.read_csv(tiles_data_file)

    data_transform_img = transforms.Compose([
        #transforms.RandomVerticalFlip(p=VP),
        #transforms.RandomHorizontalFlip(p=HP),
        #transforms.RandomRotation([30, 90]),
        #transforms.ToTensor(),
        #transforms.RandomApply([AddGaussianNoise(0, 1)], p=0.5),
        #transforms.RandomResizedCrop(size=(64, 64)),
        #transforms.RandomApply([transforms.RandomResizedCrop(size=(64, 64))], p=0.1),
        transforms.RandomApply([transforms.RandomRotation([3, 5])], p=0.5),
        transforms.Normalize(mean=[0],std=[1])
        ])

    data_transform_msk = transforms.Compose([
        #transforms.RandomVerticalFlip(p=VP),
        #transforms.RandomHorizontalFlip(p=HP),
        #transforms.RandomRotation([30, 90]),
        #transforms.ToTensor(),
        #transforms.RandomResizedCrop(size=(64, 64)),
        #transforms.RandomApply([transforms.RandomResizedCrop(size=(64, 64))], p=0.5),
        transforms.RandomApply([AddGaussianNoise(0, 1)], p=0.5),
        #transforms.Normalize(mean=[0],std=[1])
        ])

    dataloaders = get_dataloaders(tiles_data,
                                batch_size,
                                num_workers,
                                images_dir,
                                masks_dir,
                                data_transform_img,
                                data_transform_msk,
                                )

    # UNET
    #model = un.UNET(in_channels=1, out_channels= 1)

    # UNET++
    model = arc.NestedUNet(1, input_channels=1, deep_supervision=False)

    # SEGNET
    #model = seg.SegNet(in_chn=1, out_chn= 1)

    criterion = DiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay= WD)

    model.to(device)
    criterion.to(device)

    for epoch in range(1, n_epochs + 1):
        print("\nEpoch {}/{} {}".format(epoch, n_epochs, time.strftime("%Y/%m/%d-%H:%M:%S")))
        print("-" * 10)

        train_metrics = train(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            device
        )
        val_metrics = evaluate(
            model,
            dataloaders['test'],
            criterion,
            device
        )

        # mask_data = np.array([[1, 2, 2, ..., 2, 2, 1], ...])
        class_labels = {
            0: "land",
            1: "water",
        }

        sar_image, predicted_image, ndwi_image, iou = evaluate_single_image(model, tiles_data, images_dir, masks_dir, device)
        int_sar_image = np.array(predicted_image > 0.5).astype(int)
        ndwi_image = ndwi_image[0]
        int_ndwi_image = np.array(ndwi_image > 0.5).astype(int)
        mask_img = wandb.Image(sar_image, masks={
            "predictions": {
                "mask_data": int_sar_image,
                "class_labels": class_labels
            },
            "ndwi": {
                "mask_data": int_ndwi_image,
                "class_labels": class_labels
            }
        }, caption=["Water detection", "fd", "fds"])

        predicted_image = wandb.Image(predicted_image, caption="Predicted image")
        mask_img = wandb.Image(mask_img, caption="Mask image")

        metrics = {
            **train_metrics, **val_metrics, 'prediction': predicted_image,
            'mask': mask_img
        }

        print('Train loss: {}, Val loss: {}'.format(metrics['train_loss'], metrics['val_loss']))
        wandb.log(metrics)

    model_name = os.getenv("MODEL_NAME")
    model_file = f'{run_name}_{model_name}.pth'
    save_model(model, model_dir, model_file)

    evaluate_single_image(model, tiles_data, images_dir, masks_dir, device)

    wandb.finish()

def main():

    load_dotenv()
    config = dotenv_values()

    LR = 0.0001
    BS = 256
    WD = 0.001

    full_cycle(BS, LR, WD)


# start = time.time()
main()
# end = time.time()
# total_time = end - start
# print("%s: Total time = %f seconds" % (time.strftime("%Y/%m/%d-%H:%M:%S"), total_time))
