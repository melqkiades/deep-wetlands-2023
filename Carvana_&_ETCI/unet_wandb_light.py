# -*- coding: utf-8 -*-
"""Unet_wandb_GENERAL_light.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18ISyBOqguy2b4cXSjlqxH9XOwl4GBEvQ
"""

#from google.colab import drive
#drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd drive/MyDrive/Colab Notebooks/Unet

#!ls

#!pip install wandb

#!pip install torchmetrics

import wandb

wandb.login()

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchmetrics
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import os
import random
import numpy as np
from torchmetrics.classification import BinaryJaccardIndex
from PIL import Image,ImageFile
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder

from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from torchvision.transforms import Compose, ToTensor, Resize
#import calc_iou as c

#Solution to corrupted png files: source:https://github.com/python-pillow/Pillow/issues/5631
ImageFile.LOAD_TRUNCATED_IMAGES = True 

#from torchvision.datasets import ImageFolder
#from torch.utils.data import Subset
#from sklearn.model_selection import train_test_split
#from torchvision.transforms import Compose, ToTensor, Resize

def split_dataset(dataset, split=0.5):
    val_idx, test_idx = train_test_split(list(range(len(dataset))), test_size= split)
    datasets = {}
    datasets['val'] = Subset(dataset, val_idx)
    datasets['test'] = Subset(dataset, test_idx)
    return datasets

"""dataset = ImageFolder('C:\Datasets\lcms-dataset', transform=Compose([Resize((224,224)),ToTensor()]))
print(len(dataset))
datasets = train_val_dataset(dataset)
print(len(datasets['train']))
print(len(datasets['val']))
# The original dataset is available in the Subset class
print(datasets['train'].dataset)

dataloaders = {x:DataLoader(datasets[x],32, shuffle=True, num_workers=4) for x in ['train','val']}
x,y = next(iter(dataloaders['train']))
print(x.shape, y.shape)"""

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# PREPARATION OF CARVANA DATASET
class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        #if '(1)' in self.images[index]:
        #  print ('\nBINGO', self.images[index])
        #  self.images[index] = self.images[index].replace(" (1).jpg",".jpg")


        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))

       #print(mask_path)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

#prova = CarvanaDataset(
#        image_dir="data/train_images/",
#        mask_dir="data/train_masks/"
#    )

#print(len(prova))
#print(len(prova)/16)

# PREPARATION OF ETCI DATASET
class ETCI(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("_vh.png", ".png"))
        #mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))

       #print(mask_path)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        #print ("MIN mask", min(mask.all()))
        print("mask", mask)
        print()
        
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

#prova = ETCI(image_dir="ETCI/train_images/", mask_dir="ETCI/train_masks/")

#print(len(prova))
#print(len(prova)/18)

# PREPARATION OF CVPR DATASET
class CVPR(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("_vh.png", ".png"))
        #mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))

       #print(mask_path)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

#prova = CVPR(image_dir="ETCI/train_images/", mask_dir="ETCI/train_masks/")

#print(len(prova))
#print(len(prova)/18)

#UTILITY FUNCTIONS
def split_dataset(dataset, split=0.2):
    val_idx, test_idx = train_test_split(list(range(len(dataset))), test_size= split)
    datasets = {}
    datasets['train'] = Subset(dataset, val_idx)
    datasets['val'] = Subset(dataset, test_idx)
    return datasets


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    #test_dir,
    #test_maskdir,
    #test_transform,
    num_workers=4,
    pin_memory=True,
):
    #train_ds = CarvanaDataset(
#    train_ds = ETCI(
#        image_dir=train_dir,
#        mask_dir=train_maskdir,
#        transform=train_transform,
#    )
#    dsets = split_dataset(train_ds, split=0.15)
   # t_ds = dsets['train']
   # v_ds = dsets['val']
    val_ds = ETCI(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    dsets_2 = split_dataset(val_ds, split=0.15)

#    t_ds = torch.utils.data.ConcatDataset([dsets['train'], dsets_2 ['train']])
#    v_ds = torch.utils.data.ConcatDataset([dsets['val'], dsets_2 ['val']])

    t_ds = dsets_2['train']
    v_ds = dsets_2['val']

    train_loader = DataLoader(
       # train_ds,
        t_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    #val_ds = CarvanaDataset(
#    val_ds = ETCI(
#        image_dir=val_dir,
#        mask_dir=val_maskdir,
#        transform=val_transform,
#    )

    val_loader = DataLoader(
    #    val_ds,
        v_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    """test_ds = ETCI(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )"""


    return train_loader, val_loader#, test_loader

def validation_loss(loader, model, loss_fn):

    #wandb.watch(model, loss_fn, log="all", log_freq=10)

    with torch.no_grad():

      for (data, targets) in loader:
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            #wandb.log({"epoch": epoch, "train_loss": loss}, step = len(data))
            ##wandb.log({"Val_Loss": loss})

    return loss

def computeIOU(output, target):
  output = torch.argmax(output, dim=1).flatten()
  target = target.flatten()

  no_ignore = target.ne(255).cuda()
  output = output.masked_select(no_ignore)
  target = target.masked_select(no_ignore)
  intersection = torch.sum(output * target)
  union = torch.sum(target) + torch.sum(output) - intersection
  iou = (intersection + .0000001) / (union + .0000001)

  if iou != iou:
    print("failed, replacing with 0")
    iou = torch.tensor(0).float()

  return iou


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            print("DIM X", x.shape)
            print("DIM Y", y.shape)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            print("DIM Preds", preds.shape)

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            IoU = BinaryJaccardIndex().to(DEVICE)
            #score = computeIOU(preds,y)
            score = IoU(preds,y)
            #score = c.get_IoU(preds, y)

            acc = num_correct/num_pixels*100
            dice = dice_score/len(loader)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")

    print(f"IoU: {score}")

    #wandb.log({"Pixel_Accuracy": num_correct/num_pixels*100, "Dice_Score": dice_score/len(loader), "IoU": IoU(preds, y)})

    model.train()

    return acc, dice, score

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        #print("TEST", len(loader))

        if idx % (len(loader)-1) == 1 and idx != 0:

          #print('Type of preds', preds.size())
          #print('Type of y', y.unsqueeze(1).size())
          #print('Len preds', len(preds))

          f, axarr = plt.subplots(1,5)
          for id, img in enumerate(preds[len(preds)-5:,:,:,:]):
            axarr[id].imshow(img.permute(1, 2, 0).numpy(force=True))
          f.tight_layout(pad=0.5)
         # f.savefig(f'{folder}/pred_{idx}.png')


          f2, axr = plt.subplots(1,5)
          for id, img in enumerate(y.unsqueeze(1)[len(preds)-5:,:,:,:]):
            axr[id].imshow(img.permute(1, 2, 0).numpy(force=True))
          f2.tight_layout(pad=0.5)
         # f2.savefig(f'{folder}/{idx}.png')

          f3, axo = plt.subplots(1,5)
          for id, img in enumerate(x[len(preds)-5:,:,:,:]):
              
              print("TEST image x size", img.shape)
            
              axo[id].imshow(img.permute(1, 2, 0).numpy(force=True))
          f3.tight_layout(pad=0.5)
         # f3.savefig(f'{folder}/orig_{idx}.png')

         # out_path = os.path.join(folder,"pred_"+str(idx)+".png")
         # out_img = Image.open(out_path)

         # in_path = os.path.join(folder,str(idx)+".png")
         # in_img = Image.open(in_path)

         # orig_path = os.path.join(folder,"orig_"+str(idx)+".png")
         # orig = Image.open(orig_path)

    model.train()

   # return out_img, in_img, orig
    return f, f2, f3

# BUILDING UNET MODEl (Source: https://www.classcentral.com/course/youtube-pytorch-image-segmentation-tutorial-with-u-net-everything-from-scratch-baby-126811)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
           # nn.Dropout(p=0.2)
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape
    print('OK')

test()

#TRAIN U-NET
def train_fn(loader, model, optimizer, loss_fn, scaler, epoch,scheduler):

    #wandb.watch(model, loss_fn, log="all", log_freq=10)

    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            #wandb.log({"epoch": epoch, "id": batch_idx, "train_loss": loss}, step = len(data))
            #wandb.log({"epoch": epoch, "id": batch_idx, "Train_Loss": loss})

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    scheduler.step()
    lr = scheduler.get_lr()

    return loss, lr

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE =25 #other 1_R# 15 mixed regions#22 1_R #18 2_R # C 16
NUM_EPOCHS = 1
NUM_WORKERS = 4
IMAGE_HEIGHT = 256 # C 160  # 1280 originally
IMAGE_WIDTH = 256 # C 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ETCI/train_images/" # "data/train_images/"
TRAIN_MASK_DIR = "/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ETCI/train_masks/" # "data/train_masks/"
VAL_IMG_DIR = "/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ETCI/val_images/" # "data/val_images/"
VAL_MASK_DIR = "/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ETCI/val_masks/" # "data/val_masks/"
#TEST_IMG_DIR = "/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ETCI/test_images/" # "data/test_images/"
#TEST_MASK_DIR = "/mimer/NOBACKUP/groups/deep-wetlands-2023/Ezio/ETCI/test_masks/" # "data/test_masks/"
REG ="Annealing"# "Warm_Anneal"#"Linear" #"Annealing"

def main():

  wandb.init(
    # set the wandb project where this run will be logged
    project="U-Net_improved",
    # track hyperparameters and run metadata
    config = dict(
    epochs= NUM_EPOCHS,
    batch_size= BATCH_SIZE,
    learning_rate= LEARNING_RATE,
    regularization = REG,
    #dataset="Carvana",
    dataset="ETCI",
    architecture="U-NET",
    platform = "Alvis")
  )

  train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

  val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

  #test_transforms = val_transforms

  model = UNET(in_channels=3, out_channels=1).to(DEVICE)
  loss_fn = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

  #train_loader, val_loader, test_loader = get_loaders(
  train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        #TEST_IMG_DIR,
        #TEST_MASK_DIR,
        #test_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
  
  T_max = len(train_loader)
  #T_0 = int(T_max/2)

  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)
 # scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1.0, total_iters=T_max, last_epoch=- 1, verbose=False)
 # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
 #                                       T_0 = 20,# Number of iterations for the first restart
 #                                       T_mult = 1, # A factor increases TiTi​ after a restart
                                       # eta_min = LEARNING_RATE) # Minimum learning rate
 #                                      )

  if LOAD_MODEL:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

  check_accuracy(val_loader, model, device=DEVICE)
  scaler = torch.cuda.amp.GradScaler()

  #wandb.watch(model, loss_fn, log="all", log_freq=10)

  epochs = [i for i in range (NUM_EPOCHS)]

  Train_L = []
  Train_A = []
  Train_I = []

  Val_L = []
  Val_A = []
  Val_I = []

  for epoch in range(NUM_EPOCHS):
    train_loss, lr = train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch,scheduler)
    train_acc, train_dice, train_IoU = check_accuracy(train_loader, model, device=DEVICE)

    #checkpoint = {
    #        "state_dict": model.state_dict(),
    #        "optimizer":optimizer.state_dict(),
    #    }


    #save_checkpoint(checkpoint)
    val_loss = validation_loss(val_loader, model, loss_fn)
    val_acc, val_dice, val_IoU = check_accuracy(val_loader, model, device=DEVICE)

      # print some examples to a folder
    pred_img, in_img, orig = save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)

    image_in = wandb.Image(in_img, caption="Ground_Truth")
    image_out = wandb.Image(pred_img, caption="Predicted_mask")
    original = wandb.Image(orig, caption="Input_Image")

    Train_L.append(train_loss)
    Train_A.append(train_acc)
    Train_I.append(train_IoU)

    Val_L.append(val_loss)
    Val_A.append(val_acc)
    Val_I.append(val_IoU)

    wandb.log({"Predicted_Mask": image_out,"Ground Truth": image_in, "Input_Image": original,
               "Train_Loss": train_loss, "Train_Acc": train_acc, "Train_IoU": train_IoU,
               "Val_Loss": val_loss, "Val_Acc": val_acc, "Val_IoU": val_IoU,"Learning_Rate": lr[0],
               "epoch": epoch})

    # save model every 20 epoch
    if epoch % 20 == 0:
      checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
      save_checkpoint(checkpoint)


  #torch.onnx.export(model, x, "unet.onnx")
  #wandb.save("unet.onnx")

  wandb.log({"Loss_Plot" : wandb.plot.line_series(
                      xs= epochs,
                      ys=[Train_L, Val_L],
                      keys=["Train", "Val"],
                      title="Train_and_Val_Loss",
                      xname="epochs")})

  wandb.log({"Acc_Plot" : wandb.plot.line_series(
                      xs= epochs,
                      ys=[Train_A, Val_A],
                      keys=["Train", "Val"],
                      title="Train_and_Val_Pixel_Accuracy",
                      xname="epochs")})

  wandb.log({"IoU_Plot" : wandb.plot.line_series(
                      xs= epochs,
                      ys=[Train_I, Val_I],
                      keys=["Train", "Val"],
                      title="Train_and_Val_IoU",
                      xname="epochs")})

  wandb.finish()

  #save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)

if __name__ == "__main__":
    main()

