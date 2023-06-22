from train import (
    train,
    final_evaluation
)
from model import Autoencoder
from hype_tune import hype_tune
from utils import get_loaders
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
import torch.optim as optim
import torch.nn as nn

# -------------- HYPERPARAMETERS -----------------
LR = 0.00001
WEIGHT_DECAY = 0.01
RP = 0.0
HP = 0.0
VP = 0.0
KERNEL_SIZE = 3
DEPTH = 3
BOTTLENECK_SIZE = 32
BATCH_SIZE = 128
NUM_EPOCHS = 1
NUM_WORKERS = 8
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
PIN_MEMORY = True
SAVE_MODEL = True
PLOT = True
LOCAL = True
HYPE_TUNE = False
# ----------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if LOCAL:
    TRAIN_IMG_DIR = 'Input'
    TRAIN_MASK_DIR = 'Target'
    VAL_IMG_DIR = 'Input'
    VAL_MASK_DIR = 'Target'
    TEST_IMG_DIR = 'Input'
    TEST_MASK_DIR = 'Target'
else:
    TRAIN_IMG_DIR = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Johanna/SAR_dataset/SAR/train/data'
    TRAIN_MASK_DIR = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Johanna/SAR_dataset/Annotations/train/data'
    VAL_IMG_DIR = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Johanna/SAR_dataset/SAR/val/data'
    VAL_MASK_DIR = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Johanna/SAR_dataset/Annotations/val/data'
    TEST_IMG_DIR = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Johanna/SAR_dataset/SAR/test/data'
    TEST_MASK_DIR = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Johanna/SAR_dataset/Annotations/test/data'

FILENAME = 'Pretrain_Autoencoder_VH_'
HYPE_NAME = 'Pretrain Conv Autoencoder sweep'
TAGS = ['Conv Autoencoder', 'Only VH', 'New dataset', 'pretraining']
THREE_CHANNELS = False
MODEL = 'convauto'
PRETRAIN = True


def main(model):
    if not LOCAL:
        wandb.init(project="Pre-train Autoencoder", config={
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            'prob_horizontal_flip': HP,
            'prob_vertical_flip': VP,
            'prob_rotation': RP,
            'kernel_size': KERNEL_SIZE,
            'depth': DEPTH,
            'bottleneck_size': BOTTLENECK_SIZE,
            'loss_function': 'Dice',
            "train_transform": 'Random vertical flip',
        }, tags=TAGS)


    train_transform = A.Compose(
        [
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            ToTensorV2(),
        ]
    )

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_loader, val_loader, test_loader = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, TEST_IMG_DIR, TEST_MASK_DIR, BATCH_SIZE, PIN_MEMORY, NUM_WORKERS, train_transform, val_transform, THREE_CHANNELS)
    scaler = torch.cuda.amp.GradScaler()

    model = train(model, train_loader, val_loader, loss_fn, optimizer, scaler, NUM_EPOCHS, DEVICE, FILENAME, SAVE_MODEL, LOCAL, PLOT, THREE_CHANNELS, PRETRAIN)
    final_evaluation(model, test_loader, loss_fn, DEVICE, TEST_MASK_DIR, LOCAL, THREE_CHANNELS)


if __name__ == '__main__':
    model = Autoencoder(input_size=1, depth=DEPTH, bottleneck_size=BOTTLENECK_SIZE,
                              output_size=1, kernel_size=KERNEL_SIZE).to(DEVICE)
    if not HYPE_TUNE:
        main(model)
    else:
        hype_tune(HYPE_NAME, TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, PIN_MEMORY, NUM_WORKERS, DEVICE, TAGS, THREE_CHANNELS, MODEL, PRETRAIN)

