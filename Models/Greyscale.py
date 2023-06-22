from train import (
    train,
    final_evaluation,
    DiceLoss
)
from model import UNet
from hype_tune import hype_tune
from utils import get_loaders
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
import torch.optim as optim

#-------------- HYPERPARAMETERS -----------------
LR = 0.000005
HP = 0.0
RP = 0.2
VP = 0.4
WEIGHT_DECAY = 0.1
BATCH_SIZE = 128
NUM_EPOCHS = 1
NUM_WORKERS = 8
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
PIN_MEMORY = True
SAVE_MODEL = False
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
    #TEST_IMG_DIR = 'Input'
    #TEST_MASK_DIR = 'Target'
    TEST_IMG_DIR = '../../../../Downloads/SAR_norm_dataset/SAR/test/data'
    TEST_MASK_DIR = '../../../../Downloads/SAR_norm_dataset/Annotations/test/data'
else:
    TRAIN_IMG_DIR = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Johanna/SAR_norm_dataset/SAR/train/data'
    TRAIN_MASK_DIR = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Johanna/SAR_norm_dataset/Annotations/train/data'
    VAL_IMG_DIR = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Johanna/SAR_norm_dataset/SAR/val/data'
    VAL_MASK_DIR = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Johanna/SAR_norm_dataset/Annotations/val/data'
    TEST_IMG_DIR = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Johanna/SAR_norm_dataset/SAR/test/data'
    TEST_MASK_DIR = '/mimer/NOBACKUP/groups/deep-wetlands-2023/Johanna/SAR_norm_dataset/Annotations/test/data'

FILENAME = 'Greyscale_VH_on_UNet'
HYPE_NAME = 'Greyscale sweep'
TAGS = ['Greyscale', 'Only VH', 'New dataset', 'Finetuning']
THREE_CHANNELS = True
MODEL = 'greyscale'


def main(model):
    if not LOCAL:
        wandb.init(project="Test", config={
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            'prob_horizontal_flip': HP,
            'prob_vertical_flip': VP,
            'prob_rotation': RP,
            'loss_function': 'Dice',
            "train_transform": 'Random vertical flip and rotation',
        }, tags=TAGS)


    train_transform = A.Compose(
        [
            A.Rotate(limit=35, p=RP),
            A.HorizontalFlip(p=HP),
            A.VerticalFlip(p=VP),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ]
    )

    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_loader, val_loader, test_loader = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, TEST_IMG_DIR, TEST_MASK_DIR, BATCH_SIZE, PIN_MEMORY, NUM_WORKERS, train_transform, val_transform, THREE_CHANNELS)
    scaler = torch.cuda.amp.GradScaler()

    #model = train(model, train_loader, val_loader, loss_fn, optimizer, scaler, NUM_EPOCHS, DEVICE, FILENAME, SAVE_MODEL, LOCAL, PLOT, THREE_CHANNELS)
    final_evaluation(model, test_loader, loss_fn, DEVICE, TEST_MASK_DIR, LOCAL, THREE_CHANNELS)


if __name__ == '__main__':
    model = UNet().to(DEVICE)
    if LOCAL:
        model.load_state_dict(torch.load('../../checkpoint', map_location=torch.device('cpu'))['state_dict'])
    else:
        model.load_state_dict(torch.load('../Old_Code/U-Net_on_Coco_50_epochs_1')['state_dict'])

    if not HYPE_TUNE:
        main(model)
    else:
        hype_tune(HYPE_NAME, TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, PIN_MEMORY, NUM_WORKERS, DEVICE, TAGS, THREE_CHANNELS, MODEL)

