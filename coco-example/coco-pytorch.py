import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import wandb
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms, datasets
from functools import partial
from time import time
from PIL import Image
import io


class Unet(nn.Module):

    def __init__(self, input_dim=572, num_input_channels=4, num_output_channels=2, num_first_level_channels=64,
                depth=5, conv_kernel_size=3, conv_stride=1, max_pool_size=2,
                up_conv_kernel_size=2, up_conv_stride=2, padding=False):
        super().__init__()

        self.input_dim = input_dim
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.num_first_level_channels = num_first_level_channels
        self.depth = depth
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.max_pool_size = max_pool_size
        self.up_conv_kernel_size = up_conv_kernel_size
        self.up_conv_stride = up_conv_stride
        self.padding = padding
        self.depth_num_channels = [self.num_first_level_channels]
        for i in range(self.depth-1):
            self.depth_num_channels.append(self.depth_num_channels[-1]*2)
        if not self.padding:
            self.encoder_skip_dims = [self.input_dim-2*(self.conv_kernel_size-1)]
            for i in range(1, len(self.depth_num_channels) - 1):
                self.encoder_skip_dims.append(self.encoder_skip_dims[-1] // self.max_pool_size - 2 * (self.conv_kernel_size - 1))
            self.decoder_skip_dims = [(self.encoder_skip_dims[-1] // self.max_pool_size - 2 * (self.conv_kernel_size - 1)) * 2]
            for i in range(1, len(self.depth_num_channels) - 1):
                self.decoder_skip_dims.insert(0, (self.decoder_skip_dims[0] - 2 * (self.conv_kernel_size - 1)) * self.up_conv_stride)
        self.encoder_layers = nn.ModuleList([self.unet_block(num_input_channels, self.depth_num_channels[0]), nn.MaxPool2d(2)])
        for i in range(len(self.depth_num_channels) - 2):
            self.encoder_layers.append(
                self.unet_block(self.depth_num_channels[i], self.depth_num_channels[i + 1]))
            self.encoder_layers.append(nn.MaxPool2d(2))
        self.bottleneck = self.unet_block(self.depth_num_channels[-2], self.depth_num_channels[-1])
        self.decoder_layers = nn.ModuleList([
            nn.ConvTranspose2d(self.depth_num_channels[-1], self.depth_num_channels[-2], self.up_conv_kernel_size,
                               self.up_conv_stride),
            self.unet_block(self.depth_num_channels[-1], self.depth_num_channels[-2])])
        for i in range(len(self.depth_num_channels) - 2, 0, -1):
            self.decoder_layers.append(nn.ConvTranspose2d(self.depth_num_channels[i], self.depth_num_channels[i - 1],
                                                          self.up_conv_kernel_size, self.up_conv_stride))
            self.decoder_layers.append(
                self.unet_block(self.depth_num_channels[i], self.depth_num_channels[i - 1]))
        if not self.padding:
            self.crops = []
            for i in range(self.depth - 1):
                self.crops.append(transforms.CenterCrop(self.decoder_skip_dims[i]))
        self.final_layer = nn.Conv2d(self.depth_num_channels[0], self.num_output_channels, 1)

    def forward(self, x):
        skip_connections = []
        for i in range(0, len(self.encoder_layers), 2):
            x = self.encoder_layers[i](x)
            if not self.padding:
                skip_connections.insert(0, self.crops[i // 2](x))
            else:
                skip_connections.insert(0, x)
            x = self.encoder_layers[i + 1](x)

        x = self.bottleneck(x)

        for i in range(0, len(self.decoder_layers), 2):
            x = self.decoder_layers[i](x)
            x = torch.cat([skip_connections[i // 2], x], 1)
            x = self.decoder_layers[i + 1](x)
        x = torch.nn.Softmax(dim=1)(self.final_layer(x))

        return x

    def conv_block(self, num_in_channels, num_out_channels):
        if not self.padding:
            padding_arg = 'valid'
        else:
            padding_arg = 'same'
        conv_block = nn.Sequential(nn.Conv2d(num_in_channels, num_out_channels, self.conv_kernel_size, self.conv_stride,
                                             padding=padding_arg),
                                   nn.BatchNorm2d(num_out_channels),
                                   nn.ReLU())
        return conv_block


    def unet_block(self, num_in_channels, num_out_channels):
        unet_block = nn.Sequential(self.conv_block(num_in_channels, num_out_channels),
                                   self.conv_block(num_out_channels, num_out_channels))
        return unet_block



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
    
def intersection_over_union(y_pred, y_true):

    smooth = 1e-6
    y_pred = y_pred[:, 0].view(-1) > 0.5
    y_true = y_true[:, 0].view(-1) > 0.5
    intersection = (y_pred & y_true).sum() + smooth
    union = (y_pred | y_true).sum() + smooth
    iou = intersection / union
    return iou

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def get_mask(image, annotation, coco_annotations, augment):
    image = transforms.ToTensor()(image)
    mask = torch.zeros((image.shape[1],image.shape[2]))
    if augment:
        params = random_crop.get_params(image, scale=(0.5, 1.0), ratio=(0.75, 1.33))
        image = transforms.functional.crop(image, *params)
    image = transforms.Resize((256, 256))(image)
    image = normalization(image)
    for i in range(len(annotation)):
#         mask = np.maximum(coco.annToMask(annotation[i])*annotation[i]["category_id"], mask)
        mask = torch.maximum(torch.from_numpy(coco_annotations.annToMask(annotation[i])), mask)
    mask = mask[None,:,:]
    if augment:
        mask = transforms.functional.crop(mask, *params)
    mask = transforms.Resize((256, 256))(mask)
    return image, mask


invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

input_dim = 256
num_input_channels = 3
num_output_channels = 1
batch_size = 32
num_workers = 8

coco_train = COCO("/mimer/NOBACKUP/Datasets/Microsoft-COCO/annotations/instances_train2017.json")
coco_val = COCO("/mimer/NOBACKUP/Datasets/Microsoft-COCO/annotations/instances_val2017.json")

random_crop = transforms.RandomResizedCrop(256)
normalization = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

datasets = {'train':datasets.CocoDetection("/mimer/NOBACKUP/Datasets/Microsoft-COCO/train2017",
                                 "/mimer/NOBACKUP/Datasets/Microsoft-COCO/annotations/instances_train2017.json",transforms=partial(get_mask, coco_annotations=coco_train, augment=True)),
            'val':datasets.CocoDetection("/mimer/NOBACKUP/Datasets/Microsoft-COCO/val2017",
                                 "/mimer/NOBACKUP/Datasets/Microsoft-COCO/annotations/instances_val2017.json",transforms=partial(get_mask, coco_annotations=coco_val, augment=False))}

dataloaders = {'train':DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
               'val':DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers)}

device = torch.device("cuda:0")


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    loss_sum = 0.
    iou_sum = 0.
    
    print('Train:')
    for i_batch, (image, true_seg) in enumerate(dataloader):
        print('\r' + f'Batch: {i_batch}/{len(dataloader)}', end='')
        image = image.to(device)
        true_seg = true_seg.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            model_seg = model(image)

            loss = criterion(model_seg, true_seg)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            iou_sum += intersection_over_union(model_seg, true_seg)

    avg_loss = loss_sum / (i_batch + 1)
    avg_iou = iou_sum / (i_batch + 1)
    print(f" Loss: {avg_loss}, :IoU: {avg_iou}")

    metrics = {
        'train_loss': avg_loss, 'train_iou': avg_iou
    }

    return metrics

def evaluate(model, dataloader, criterion, device, plot_result=True):
    model.eval()
    loss_sum = 0.
    iou_sum = 0.
    print('Validation:')
    for i_batch, (image, true_seg) in enumerate(dataloader):
        print('\r' + f'Batch: {i_batch}/{len(dataloader)}', end='')
        image = image.to(device)
        true_seg = true_seg.to(device)

        with torch.set_grad_enabled(False):
            model_seg = model(image)

            loss = criterion(model_seg, true_seg)

            loss_sum += loss.item()
            iou_sum += intersection_over_union(model_seg, true_seg)

    if plot_result:
        images = invTrans(image.cpu()).numpy()
        true_segs = true_seg.cpu().numpy()
        model_segs = (model_seg.cpu().numpy() > 0.5).astype(float)

        fig, axs = plt.subplots(5, 3, figsize=(15, 15))
        for i in range(5):
            image = np.transpose(images[i], [1, 2, 0])
            true_seg = np.transpose(true_segs[i], [1, 2, 0])
            model_seg = np.transpose(model_segs[i], [1, 2, 0])
            axs[i, 0].imshow(image)
            axs[i, 0].set_title('Image')
            axs[i, 1].imshow(true_seg)
            axs[i, 1].set_title('True segmentation map')
            axs[i, 2].imshow(model_seg)
            axs[i, 2].set_title('Predicted segmentation map')
        fig.tight_layout()
        fig.canvas.draw()
        wandb_fig = plt.gcf()
        wandb_img = wandb.Image(fig2img(wandb_fig))
        
                  
    avg_loss = loss_sum / (i_batch + 1)
    avg_iou = iou_sum / (i_batch + 1)
    print(f" Loss: {avg_loss}, :IoU: {avg_iou}")

    metrics = {
        'val_loss': avg_loss, 'val_iou': avg_iou, 'examples':wandb_img
    }

    return metrics


wandb.login(key='####')
num_first_level_channels = 64
depth = 4
model = Unet(input_dim, num_input_channels, num_output_channels, num_first_level_channels, depth, padding=True)
model.to(device)
lr = 0.0004863
weight_decay = 0.0003204
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_func = DiceLoss()


wandb.init(project="coco-unet", config={
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "depth":depth,
    })
n_epochs = 100
for epoch in range(1, n_epochs + 1):
    starttime = time()
    print("\nEpoch {}/{}".format(epoch, n_epochs))
    print("-" * 10)

    train_metrics = train(
        model,
        dataloaders["train"],
        loss_func,
        optimizer,
        device
    )
    traintime = time()
    print(f'Train duration: {traintime - starttime} s.')
    val_metrics = evaluate(
        model,
        dataloaders['val'],
        loss_func,
        device,
        plot_result=True
    )
    valtime = time()
    print(f'Val duration: {valtime - traintime} s.')
    metrics = {**train_metrics,  **val_metrics}

    torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, "checkpoint.pt")
    wandb.log(metrics)

