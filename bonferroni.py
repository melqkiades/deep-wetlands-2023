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

import train_models as tr
import archs as arc
import SegNet as seg
import unet as un

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


def load_model_func( arch ,model_file, device):

    if arch  == 'UNET':
        loaded_model = UNET(in_channels=1, out_channels=1)

    if arch  =='UNET++':
        loaded_model = arc.NestedUNet(1, input_channels=1, deep_supervision=False)#U-NET++

    if arch == 'SegNet':
        loaded_model = seg.SegNet(in_chn=1, out_chn= 1)

    loaded_model.to(device)
    loaded_model.load_state_dict(torch.load(model_file, map_location=device))
    loaded_model.eval()

    print('Model file {} successfully loaded.'.format(model_file))

    return loaded_model


def evaluate_images(UN,UNp,SEG, tiles_data, images_dir, ndwi_masks_dir, device):

    IoU_UN = []
    IoU_UNp = []
    IoU_SEG = []

    for i in range(len(tiles_data)):

        UN.eval()
        UNp.eval()
        SEG.eval()

        index = tiles_data.iloc[i]['id']
        image_path = images_dir + str(index) + '-sar.tif'

        sar_image = rio.open(image_path).read()
        sar_image = np.expand_dims(sar_image[0,:,:],axis=0)

        ndwi_image_path = ndwi_masks_dir + str(index) + '-mask.tif'
        ndwi_image = rio.open(ndwi_image_path).read()

        batch_sar_image = sar_image[None, :]
        batch_sar_image = torch.from_numpy(batch_sar_image.astype(np.float32)).to(device)

        pred_image_UN = UN(batch_sar_image).cpu().detach().numpy()
        # UNET++ with deep supervision = True
        #pred_image_UNp = UNp(batch_sar_image)[-1].cpu().detach().numpy()
        # UNET++ with deep supervision = False
        pred_image_UNp = UNp(batch_sar_image).cpu().detach().numpy()
        pred_image_SEG = SEG(batch_sar_image).cpu().detach().numpy()

        pred_image_UN = pred_image_UN.squeeze()
        pred_image_UNp = pred_image_UNp.squeeze()
        pred_image_SEG = pred_image_SEG.squeeze()

        iou_UN = calculate_intersection_over_union(ndwi_image[0], pred_image_UN)
        iou_UNp = calculate_intersection_over_union(ndwi_image[0], pred_image_UNp)
        iou_SEG = calculate_intersection_over_union(ndwi_image[0], pred_image_SEG)

        IoU_UN.append(iou_UN)
        IoU_UNp.append(iou_UNp)
        IoU_SEG.append(iou_SEG)

    #return np.mean(IoU), np.std(IoU), np.mean(IoU_Otsu), np.std(IoU_Otsu), np.mean(np.subtract(IoU,IoU_Otsu)), np.std(np.subtract(IoU,IoU_Otsu))
    return np.mean(IoU_UN),np.std(IoU_UN),np.mean(IoU_UNp),np.std(IoU_UNp), np.mean(IoU_SEG),np.std(IoU_SEG),np.mean(np.subtract(IoU_UN,IoU_UNp)),np.std(np.subtract(IoU_UN,IoU_UNp)),np.mean(np.subtract(IoU_UN,IoU_SEG)),np.std(np.subtract(IoU_UN,IoU_SEG)),np.mean(np.subtract(IoU_UNp,IoU_SEG)),np.std(np.subtract(IoU_UNp,IoU_SEG))


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

  best_UN = os.getenv('BEST_UN')
  best_SEG = os.getenv('BEST_SEG')
  best_UN_plus = os.getenv('BEST_UN_PLUS')
  tiles_data_file = os.getenv('TILES_DATA_FILE')

  date ='2020-2022'

  tiles_data = pd.read_csv(tiles_data_file)
  images_dir = os.getenv('IMAGES_DIR')
  masks_dir = os.getenv('MASKS_DIR')

  wandb.init(
    # set the wandb project where this run will be logged
    project="Student_comparisons",
    # track hyperparameters and run metadata
    config = dict(
    batch_size= batch_size,
    train_source ='2020-06-23',#'2018-07-04',#'2020-06-23',
    dataset="Ramsar",
    region = "Svartadalen",
    time= date,
    )
  )

  device = utils.get_device()

  if load_model:
      UN = load_model_func('UNET',best_UN, device)
      UN_plus = load_model_func('UNET++',best_UN_plus, device)
      SEG = load_model_func('SegNet',best_SEG, device)

  #t values: (n=198, tv= 1.9721), (n=1568, tv= 1.9615), (n= 2352, tv= 1.961)

  iou_UN_m,iou_UN_sd, iou_UNp_m, iou_UNp_sd, iou_SEG_m, iou_SEG_sd, UN_minus_UNp_m, UN_minus_UNp_sd,UN_minus_SEG_m, UN_minus_SEG_sd, UNp_minus_SEG_m, UNp_minus_SEG_sd  = evaluate_images(UN,UN_plus,SEG, tiles_data, images_dir, masks_dir, device)

#conf interval IoU_UNET
  iou_UN_lb = iou_UN_m - 1.9615 * np.divide (iou_UN_sd,np.sqrt(len(tiles_data)))

  iou_UN_ub= iou_UN_m + 1.9615 * np.divide (iou_UN_sd,np.sqrt(len(tiles_data)))

#conf interval IoU_UNET++
  iou_UNp_lb = iou_UNp_m - 1.9615 * np.divide (iou_UNp_sd,np.sqrt(len(tiles_data)))

  iou_UNp_ub= iou_UNp_m + 1.9615 * np.divide (iou_UNp_sd,np.sqrt(len(tiles_data)))

#conf interval IoU_SEGNET

  iou_SEG_lb = iou_SEG_m - 1.9615 * np.divide (iou_SEG_sd,np.sqrt(len(tiles_data)))

  iou_SEG_ub = iou_SEG_m + 1.9615 * np.divide (iou_SEG_sd,np.sqrt(len(tiles_data)))

#conf interval IoU_UNET minus IoU_UNET++ with BONFERRONI CORRECTION (alfa/3 = 0.01667) see https://www.statology.org/bonferroni-correction/ and https://atozmath.com/CONM/DistributionTables.aspx?q=t&q1=t%602%600.01667%601.813%602352&do=1

  UN_minus_UNp_lb = UN_minus_UNp_m -  2.3956 * np.divide (UN_minus_UNp_sd,np.sqrt(len(tiles_data)))

  UN_minus_UNp_ub = UN_minus_UNp_m +  2.3956 * np.divide (UN_minus_UNp_sd,np.sqrt(len(tiles_data)))


#conf interval IoU_UNET minus IoU_SEG with BONFERRONI CORRECTION

  UN_minus_SEG_lb = UN_minus_SEG_m -  2.3956 * np.divide (UN_minus_SEG_sd,np.sqrt(len(tiles_data)))

  UN_minus_SEG_ub = UN_minus_SEG_m +  2.3956 * np.divide (UN_minus_SEG_sd,np.sqrt(len(tiles_data)))


#conf interval IoU_UNET++ minus IoU_SEG with BONFERRONI CORRECTION

  UNp_minus_SEG_lb = UNp_minus_SEG_m -  2.3956 * np.divide (UNp_minus_SEG_sd,np.sqrt(len(tiles_data)))

  UNp_minus_SEG_ub = UNp_minus_SEG_m +  2.3956 * np.divide (UNp_minus_SEG_sd,np.sqrt(len(tiles_data)))


  metrics = {'IoU_lb_UN': iou_UN_lb,
            'IoU_mean_UN':iou_UN_m,
            'IoU_ub_UN': iou_UN_ub,
            'IoU_lb_UN+': iou_UNp_lb,
            'IoU_mean_UN+':iou_UNp_m,
            'IoU_ub_UN+': iou_UNp_ub,
            'IoU_lb_SEG': iou_SEG_lb,
            'IoU_mean_SEG':iou_SEG_m,
            'IoU_ub_SEG': iou_SEG_ub,
            'UN-minus_UN+_lb':  UN_minus_UNp_lb,
            'UN_minus_UN+_mean': UN_minus_UNp_m,
            'UN_minus_UN+_ub':  UN_minus_UNp_ub,
            'UN-minus_SEG_lb':  UN_minus_SEG_lb,
            'UN_minus_SEG_mean': UN_minus_SEG_m,
            'UN_minus_SEG_ub':  UN_minus_SEG_ub,
            'UN+-minus_SEG_lb':  UNp_minus_SEG_lb,
            'UN+_minus_SEG_mean': UNp_minus_SEG_m,
            'UN+_minus_SEG_ub':  UNp_minus_SEG_ub
            }

  wandb.log(metrics)
  wandb.finish()

if __name__ == "__main__":
   main()
