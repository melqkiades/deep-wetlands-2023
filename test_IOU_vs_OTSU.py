import wandb
import os
import random
import json
import time
from dotenv import load_dotenv, dotenv_values
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image,ImageFile
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio as rio
import utils
import baseline as bas

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchmetrics
import torch.optim as optim
import torchvision
from torchmetrics.classification import BinaryJaccardIndex
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import Dataset, DataLoader

import archs as arc
import unet as un
import SegNet as seg

wandb.login()

#Solution to corrupted png files: source:https://github.com/python-pillow/Pillow/issues/5631
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_intersection_over_union(prediction_image, true_image):
    smooth = 1e-6
    prediction_image = prediction_image > 0.5
    true_image = true_image > 0.5
    intersection = (prediction_image & true_image).sum() + smooth
    union = (prediction_image | true_image).sum() + smooth
    intersection_over_union = intersection / union

    return intersection_over_union


def load_model_func(model_file, device):

    #UNET
    loaded_model = un.UNET(in_channels=1, out_channels=1)

    #UNET++
    #loaded_model = arc.NestedUNet(1, input_channels=1, deep_supervision=False)#U-NET++

    #SegNet
    #loaded_model = seg.SegNet(in_chn=1, out_chn= 1)#dutiful, treasured

    loaded_model.to(device)
    loaded_model.load_state_dict(torch.load(model_file, map_location=device))
    loaded_model.eval()

    print('Model file {} successfully loaded.'.format(model_file))

    return loaded_model


def evaluate_images(model, tiles_data, images_dir, ndwi_masks_dir, device):

    IoU = []
    IoU_Otsu = []

    for i in range(len(tiles_data)):
        model.eval()

        index = tiles_data.iloc[i]['id']
        image_path = images_dir + str(index) + '-sar.tif'

        sar_image = rio.open(image_path).read()
        sar_image = np.expand_dims(sar_image[0,:,:],axis=0)

        ndwi_image_path = ndwi_masks_dir + str(index) + '-mask.tif'
        ndwi_image = rio.open(ndwi_image_path).read()

        batch_sar_image = sar_image[None, :]
        batch_sar_image = torch.from_numpy(batch_sar_image.astype(np.float32)).to(device)

        # UNET++ with deep supervision = True
        #pred_image = model(batch_sar_image)[-1].cpu().detach().numpy()

        #All other cases: UNET, SegNet and UNET++ with deep supervision = False
        pred_image = model(batch_sar_image).cpu().detach().numpy()
        pred_image = pred_image.squeeze()

        iou = calculate_intersection_over_union(ndwi_image[0], pred_image)

        otsu = bas.otsu_threshold(batch_sar_image.cpu().numpy())
        iou_otsu = calculate_intersection_over_union(ndwi_image[0], otsu)

        IoU.append(iou)
        IoU_Otsu.append(iou_otsu)

    return np.mean(IoU), np.std(IoU), np.mean(IoU_Otsu), np.std(IoU_Otsu), np.mean(np.subtract(IoU,IoU_Otsu)), np.std(np.subtract(IoU,IoU_Otsu))


def main():

  load_dotenv()

  n_epochs = int(os.getenv('EPOCHS'))
  learning_rate = float(os.getenv('LEARNING_RATE'))
  seed = int(os.getenv('RANDOM_SEED'))
  batch_size = int(os.getenv('BATCH_SIZE'))
  num_workers = int(os.getenv('NUM_WORKERS'))
  patch_size = int(os.getenv('PATCH_SIZE'))
  image_height = int(os.getenv('IMAGE_HEIGHT'))
  image_width = int(os.getenv('IMAGE_WIDTH'))
  load_model = os.getenv('LOAD_MODEL')

  model_file = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/models/SL/DA/fresh-yogurt-53_Orebro lan_mosaic_2020-06-23_sar_VH_25-epochs_0.00005-lr_42-rand.pth'#unet_da QUATER
  tiles_data_file ='/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ramsar/tiles_Vastmanlands lan_mosaic_2020-2022.csv'

  date ='2020-2022'
  tiles_data = pd.read_csv(tiles_data_file)

  images_dir = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ramsar/SAR_Imagery/Svartadalen/2020-2022/'#2018-2019 os.getenv('SAR_DIR') + '/'
  masks_dir ='/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ramsar/Annotations/Svartadalen/2020-2022/'#2018-2019 os.getenv('NDWI_MASK_DIR') + '/'

  wandb.init(
    # set the wandb project where this run will be logged
    project="ramsar_conf_intervals_CORR",
    # track hyperparameters and run metadata
    config = dict(
    #epochs= n_epochs,
    batch_size= batch_size,
    #learning_rate= learning_rate
    train_source ='2020-06-23',
    dataset="Ramsar",
    region = "Svartadalen",
    time= date,
    architecture="U-NET"
    )
  )

  device = utils.get_device()
                                     )
  if load_model:
      model = load_model_func(model_file, device)

  iou_m,iou_sd, iou_otsu_m, iou_otsu_sd, deep_minus_otsu_m, deep_minus_otsu_sd  = evaluate_images(model, tiles_data, images_dir, masks_dir, device)

  iou_lb = iou_m - 1.9615 * np.divide (iou_sd,np.sqrt(len(tiles_data)))

  iou_ub = iou_m + 1.9615 * np.divide (iou_sd,np.sqrt(len(tiles_data)))

  iou_otsu_lb = iou_otsu_m - 1.9615 * np.divide (iou_otsu_sd,np.sqrt(len(tiles_data)))

  iou_otsu_ub = iou_otsu_m + 1.9615 * np.divide (iou_otsu_sd,np.sqrt(len(tiles_data)))

  deep_minus_otsu_lb = deep_minus_otsu_m - 1.9615 * np.divide (deep_minus_otsu_sd,np.sqrt(len(tiles_data)))

  deep_minus_otsu_ub = deep_minus_otsu_m +  1.9615 * np.divide (deep_minus_otsu_sd,np.sqrt(len(tiles_data)))

  metrics = {'IoU_lb': iou_lb,
            'IoU_mean':iou_m,
            'IoU_ub': iou_ub,
            'IoU_Otsu_lb': iou_otsu_lb,
            'IoU_Otsu_mean':iou_otsu_m,
            'IoU_Otsu_ub': iou_otsu_ub,
            'Deep-minus_otsu_lb':  deep_minus_otsu_lb,
            'Deep_minus_otsu_mean': deep_minus_otsu_m,
            'Deep_minus_otsu_ub':  deep_minus_otsu_ub}

  wandb.log(metrics)
  wandb.finish()

if __name__ == "__main__":
   main()
