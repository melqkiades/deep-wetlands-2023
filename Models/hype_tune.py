import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
import torch.nn as nn
import torch.optim as optim
import torch
from utils import get_loaders
from train import (
    DiceLoss,
    train
)
from model import (
    UNet,
    ConvModel,
    AutoencoderModel,
    Autoencoder
)

TRAIN_IMG = None
TRAIN_MASK = None
VAL_IMG = None
VAL_MASK = None
BATCH_SIZE = None
PIN_MEMORY = None
NUM_WORKERS = None
DEVICE = None
TAGS = None
THREE_CHANNELS = None
MODEL_TYPE = None
PRETRAIN = None


def start(config):
    train_transform = A.Compose(
        [
            A.Rotate(limit=35, p=config.RP),
            A.HorizontalFlip(p=config.HP),
            A.VerticalFlip(p=config.VP),
            A.Normalize(
                mean=0.0,
                std=1.0,
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Normalize(
                mean=0.0,
                std=1.0,
                max_pixel_value=255,
            ),
            ToTensorV2(),
        ]
    )
    if MODEL_TYPE == 'baseline':
        model = UNet(encoder_channels=(1, 64, 128, 256, 512, 1024), decoder_channels=(1024, 512, 256, 128, 64)).to(
            DEVICE)
    elif MODEL_TYPE == 'greyscale':
        model = UNet().to(DEVICE)
        model.load_state_dict(torch.load('../Old_Code/U-Net_on_Coco_50_epochs_1')['state_dict'])
    elif MODEL_TYPE == 'conv':
        unet = UNet().to(DEVICE)
        unet.load_state_dict(torch.load('../Old_Code/U-Net_on_Coco_50_epochs_1')['state_dict'])
        model = ConvModel(unet, config.KERNEL_SIZE).to(DEVICE)
    elif MODEL_TYPE == 'convauto':
        unet = UNet().to(DEVICE)
        unet.load_state_dict(torch.load('../Old_Code/U-Net_on_Coco_50_epochs_1')['state_dict'])
        autoencoder = Autoencoder(input_size=1, depth=config.DEPTH, bottleneck_size=config.BOTTLENECK_SIZE,
                                  output_size=3, kernel_size=config.KERNEL_SIZE).to(DEVICE)
        model = AutoencoderModel(autoencoder, unet)

    if config.DICE == 0:
        loss_fn = nn.BCEWithLogitsLoss()
    elif config.DICE == 1:
        loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    train_loader, val_loader, _ = get_loaders(TRAIN_IMG, TRAIN_MASK, VAL_IMG, VAL_MASK, VAL_IMG, VAL_MASK, BATCH_SIZE, PIN_MEMORY, NUM_WORKERS, train_transform, val_transform, THREE_CHANNELS)
    scaler = torch.cuda.amp.GradScaler()

    train(model, train_loader, val_loader, loss_fn, optimizer, scaler, 50, DEVICE, FILENAME='', SAVE_MODEL=False, LOCAL = False, PLOT=True, three_channels = THREE_CHANNELS, PRETRAIN=PRETRAIN)


def main():
    wandb.init(tags=TAGS)
    start(wandb.config)


def hype_tune(name, train_img, train_mask, val_img, val_mask, batch_size, pin_memory, num_workers, device, tags, three_channels, model_type, pretrain=False):
    if model_type == 'conv':
        sweep_configuration = {
            'method': 'random',
            'name': name,
            'metric': {
                'goal': 'maximize',
                'name': 'val_iou'
            },
            'program': 'hype_tune.py',
            'parameters': {
                'WEIGHT_DECAY': {'values': [0.001, 0.01, 0.1, 0.3, 0.5]},
                'HP': {'values': [0.0, 0.3, 0.6, 1.0]},
                'VP': {'values': [0.0, 0.3, 0.6, 1.0]},
                'RP': {'values': [0.0, 0.3, 0.6, 1.0]},
                'LR': {'values': [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]},
                'KERNEL_SIZE': {'values': [3, 5, 7]},
                'DICE': {'values': [0, 1]}
            }
        }
    elif model_type == 'convauto':
        sweep_configuration = {
            'method': 'random',
            'name': name,
            'metric': {
                'goal': 'maximize',
                'name': 'val_iou'
            },
            'program': 'hype_tune.py',
            'parameters': {
                'WEIGHT_DECAY': {'values': [0.001, 0.01, 0.1, 0.3, 0.5]},
                'HP': {'values': [0.0, 0.3, 0.6, 1.0]},
                'VP': {'values': [0.0, 0.3, 0.6, 1.0]},
                'RP': {'values': [0.0, 0.3, 0.6, 1.0]},
                'LR': {'values': [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]},
                'KERNEL_SIZE': {'values': [3, 5, 7]},
                'BOTTLENECK_SIZE': {'values': [8, 16, 32, 64]},
                'DEPTH': {'values': [2, 3, 4]},
                'DICE': {'values': [0, 1]}
            }
        }
    else:
        sweep_configuration = {
            'method': 'random',
            'name': name,
            'metric': {
                'goal': 'maximize',
                'name': 'val_iou'
            },
            'program': 'hype_tune.py',
            'parameters': {
                'WEIGHT_DECAY': {'values': [0.001, 0.01, 0.1, 0.3, 0.5]},
                'HP': {'values': [0.0, 0.3, 0.6, 1.0]},
                'VP': {'values': [0.0, 0.3, 0.6, 1.0]},
                'RP': {'values': [0.0, 0.3, 0.6, 1.0]},
                'LR': {'values': [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]},
                'DICE': {'values': [0, 1]}
            }
        }
    global TRAIN_IMG, TRAIN_MASK, VAL_IMG, VAL_MASK, BATCH_SIZE, PIN_MEMORY, NUM_WORKERS, DEVICE, TAGS, THREE_CHANNELS, MODEL_TYPE, PRETRAIN
    TRAIN_IMG = train_img
    TRAIN_MASK = train_mask
    VAL_IMG = val_img
    VAL_MASK = val_mask
    BATCH_SIZE = batch_size
    PIN_MEMORY = pin_memory
    NUM_WORKERS = num_workers
    DEVICE = device
    TAGS = tags
    THREE_CHANNELS = three_channels
    MODEL_TYPE = model_type
    PRETRAIN = pretrain

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Sweeps")

    wandb.agent(sweep_id, function=main, count=20)
