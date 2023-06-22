from torch.utils.data import DataLoader
import wandb
from torchmetrics import JaccardIndex
from Dataset import SARDataset
import torchvision
import torch
from torcheval.metrics.functional import mean_squared_error


def get_loaders(train_img, train_mask, val_img, val_mask, test_img, test_mask, batch_size, pin_memory, num_workers, train_transform, val_transform, three_channels):
    train_dataset = SARDataset(train_img, train_mask, three_channels, train_transform)
    val_dataset = SARDataset(val_img, val_mask, three_channels, val_transform)
    test_dataset = SARDataset(test_img, test_mask, three_channels, val_transform)

    train_loader = DataLoader(train_dataset , batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=98, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate(model, loader, loss_function, device, three_channels, plot=True, pretrain=False):
    model.eval()
    with torch.no_grad():
        index = 0
        dice_score = 0
        loss_tot = 0
        tot_mse = 0
        jaccard = JaccardIndex(task='binary').to(device)
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.float().unsqueeze(1).to(device)

            y_pred = model(x)
            y_pred = (y_pred > 0.5).float().to(device)

            if pretrain:
                loss = loss_function(y_pred, x)
            else:
                loss = loss_function(y_pred, y)
            loss_tot += loss.item()

            if pretrain:
                tot_mse += mean_squared_error(y_pred.flatten(), x.flatten())
            else:
                dice_score += (2. * (y_pred * y).sum()) / ((y_pred + y).sum() + 1e-8)
                index += jaccard(y_pred, y)

            if plot:
                imgs = log_images(model, loader, device, three_channels, pretrain)

    avg_loss = loss_tot / len(loader)
    if pretrain:
        avg_mse = tot_mse / len(loader)
        if plot:
            wandb_log = {
                'val_loss': avg_loss, 'val_mse': avg_mse, **imgs
            }
        else:
            wandb_log = {
                'val_loss': avg_loss, 'val_mse': avg_mse
            }
    else:
        avg_iou = index / len(loader)
        dice_score = dice_score / len(loader)
        if plot:
            wandb_log = {
                'val_loss': avg_loss, 'val_iou': avg_iou, 'val_dice': dice_score, **imgs
            }
        else:
            wandb_log = {
                'val_loss': avg_loss, 'val_iou': avg_iou, 'val_dice': dice_score
            }
    model.train()

    return wandb_log


def log_images(model, loader, device, three_channels, pretrain):
    model.eval()
    for i, (x, y) in enumerate(loader):
        if i == 0:
            with torch.no_grad():
                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)
                if not pretrain:
                    y_pred = (y_pred > 0.5).float()
                y_pred = y_pred.to(device)

            length = min(len(x), 5)
            image_array = []
            for i in range(length):
                if not three_channels:
                    highest_vh = torch.max(x[i])
                    lowest_vh = torch.min(x[i])
                    vh_norm = (x[i] - lowest_vh) / (highest_vh - lowest_vh)
                    vh_norm = torch.cat((vh_norm, vh_norm, vh_norm), 0)
                else:
                    vh_norm = x[i]

                image_array.extend([vh_norm, y[i].unsqueeze(0).repeat(3, 1, 1), y_pred[i].repeat(3, 1, 1)])

            image_array = torchvision.utils.make_grid(image_array, nrow=3)
            images = wandb.Image(image_array, caption="VH, ground truth, prediction")

            wandb_log = {"examples": images}

    model.train()
    return wandb_log
