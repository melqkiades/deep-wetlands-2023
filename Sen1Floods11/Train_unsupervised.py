# The following is an example of how to utilize our Sen1Floods11 dataset for training a FCNN. In this example, we train
# and validate on hand-labeled chips of flood events. However, our dataset includes several other options that are
# detailed in the README. To replace the dataset, as outlined further below, simply replace the train, test, and
# validation split csv's, and download the corresponding dataset.
import sys

# Define model training hyperparameters
lr = 5e-4
max_epochs = 220
runname = "Sen1Floods11_unsup"
use_vv = False
dataset = "flood_hand-labeled"
batch_size = 8
num_workers = 8

# Define functions to process and augment training and testing images
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import random
from PIL import Image
import wandb
import matplotlib.pyplot as plt
import io
from matplotlib.colors import ListedColormap


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


class InMemoryDataset(torch.utils.data.Dataset):
  
    def __init__(self, data_list, preprocess_func, use_vv=False):
        self.data_list = data_list
        self.preprocess_func = preprocess_func
        self.use_vv = use_vv

    def __getitem__(self, i):
        return self.preprocess_func(self.data_list[i], self.use_vv)
  
    def __len__(self):
        return len(self.data_list)


def processAndAugment_unsupervised(data, use_vv=False):
    (x,y) = data
    im,label = x.copy(), y.copy()

    # convert to PIL for easier transforms
    if use_vv:
        im_vv = Image.fromarray(im[0])
        im_vh = Image.fromarray(im[1])
    else:
        im_vh = Image.fromarray(im[0])
    label = Image.fromarray(label.squeeze())

    # Get params for random transforms
    i, j, h, w = transforms.RandomCrop.get_params(im_vh, (256, 256))

    if use_vv:
        im_vv = F.crop(im_vv, i, j, h, w)
    im_vh = F.crop(im_vh, i, j, h, w)
    label = F.crop(label, i, j, h, w)
    if random.random() > 0.5:
        if use_vv:
            im_vv = F.hflip(im_vv)
        im_vh = F.hflip(im_vh)
        label = F.hflip(label)
    if random.random() > 0.5:
        if use_vv:
            im_vv = F.vflip(im_vv)
        im_vh = F.vflip(im_vh)
        label = F.vflip(label)
    if use_vv:
        norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])
        im = torch.stack([transforms.ToTensor()(im_vv).squeeze(), transforms.ToTensor()(im_vh).squeeze()])
    else:
        norm = transforms.Normalize([0.5235], [0.1102])
        im = transforms.ToTensor()(im_vh)
    im_aug = transforms.GaussianBlur(5, sigma=(0.1, 2.0))(im)
    im = norm(im)
    im_aug = norm(im_aug)
    label = transforms.ToTensor()(label).squeeze()
    if torch.sum(label.gt(.003) * label.lt(.004)):
        label *= 255
#   label = label.round()

    return im, im_aug, label


def processTestIm(data, use_vv=False):
    (x,y) = data
    im,label = x.copy(), y.copy()
    if use_vv:
        norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])
    else:
        norm = transforms.Normalize([0.5235], [0.1102])

    # convert to PIL for easier transforms
    if use_vv:
        im_vv = Image.fromarray(im[0]).resize((512,512))
        im_vh = Image.fromarray(im[1]).resize((512,512))
    else:
        im_vh = Image.fromarray(im[0]).resize((512, 512))
    label = Image.fromarray(label.squeeze()).resize((512,512))

    if use_vv:
        im_vvs = [F.crop(im_vv, 0, 0, 256, 256), F.crop(im_vv, 0, 256, 256, 256),
                F.crop(im_vv, 256, 0, 256, 256), F.crop(im_vv, 256, 256, 256, 256)]
    im_vhs = [F.crop(im_vh, 0, 0, 256, 256), F.crop(im_vh, 0, 256, 256, 256),
            F.crop(im_vh, 256, 0, 256, 256), F.crop(im_vh, 256, 256, 256, 256)]
    labels = [F.crop(label, 0, 0, 256, 256), F.crop(label, 0, 256, 256, 256),
            F.crop(label, 256, 0, 256, 256), F.crop(label, 256, 256, 256, 256)]

    if use_vv:
        ims = [torch.stack((transforms.ToTensor()(x).squeeze(),
                        transforms.ToTensor()(y).squeeze()))
                        for (x,y) in zip(im_vvs, im_vhs)]
    else:
        ims = [transforms.ToTensor()(x) for x in im_vhs]

    ims = [norm(im) for im in ims]
    ims = torch.stack(ims)

    labels = [(transforms.ToTensor()(label).squeeze()) for label in labels]
    labels = torch.stack(labels)

    if torch.sum(labels.gt(.003) * labels.lt(.004)):
        labels *= 255
#   labels = labels.round()
  
    return ims, labels


# Load *flood water* train, test, and validation data from splits. In this example, this is the data we will use to
# train our model.
from time import time
import csv
import os
import numpy as np
import rasterio

def getArrFlood(fname):
    return rasterio.open(fname).read()

def download_flood_water_data_from_list(l, use_vv):
    i = 0
    tot_nan = 0
    tot_good = 0
    flood_data = []
    for (im_fname, mask_fname) in l:
        if not os.path.exists(os.path.join("files/", im_fname)):
            continue
        arr_x = np.nan_to_num(getArrFlood(os.path.join("files/", im_fname)))
        if not use_vv:
            arr_x = arr_x[1:]
        arr_y = getArrFlood(os.path.join("files/", mask_fname))
        arr_y[arr_y == -1] = 255

        arr_x = np.clip(arr_x, -50, 1)
        arr_x = (arr_x + 50) / 51

        # if i % 100 == 0:
        #   print(im_fname, mask_fname)
        i += 1
        flood_data.append((arr_x,arr_y))

    return flood_data

def load_flood_data(input_root, label_root, csv_name, use_vv=True):
    files = []
    with open(csv_name) as f:
        for line in csv.reader(f):
            files.append(tuple((input_root+line[0], label_root+line[1])))

    return download_flood_water_data_from_list(files, use_vv)


if use_vv:
    unnorm = transforms.Compose([transforms.Normalize([0., 0.], [1 / 0.0820, 1 / 0.1102]),
                                 transforms.Normalize([-0.6851, -0.5235], [1., 1.])])
else:
    unnorm = transforms.Compose([transforms.Normalize([0.], [1/0.1102]),
                                transforms.Normalize([-0.5235], [1.])])


# Load training data and validation data. Note that here, we have chosen to train and validate our model on flood data.
# However, you can simply replace the load function call with one of the options defined above to load a different
# dataset.
def get_dataloaders(dataset, use_vv, batch_size, num_workers):
    if dataset == "flood_hand-labeled":
        input_root = "/mimer/NOBACKUP/groups/deep-wetlands-2023/iakovidis_data/Sen1Floods11/data/flood_events/HandLabeled/S1Hand/"
        label_root = "/mimer/NOBACKUP/groups/deep-wetlands-2023/iakovidis_data/Sen1Floods11/data/flood_events/HandLabeled/LabelHand/"
        train_csv_name = "/mimer/NOBACKUP/groups/deep-wetlands-2023/iakovidis_data/Sen1Floods11/splits/flood_handlabeled/flood_train_data.csv"
        valid_csv_name = "/mimer/NOBACKUP/groups/deep-wetlands-2023/iakovidis_data/Sen1Floods11/splits/flood_handlabeled/flood_valid_data.csv"
        test_csv_name = "/mimer/NOBACKUP/groups/deep-wetlands-2023/iakovidis_data/Sen1Floods11/splits/flood_handlabeled/flood_test_data.csv"
        bolivia_test_csv_name = "/mimer/NOBACKUP/groups/deep-wetlands-2023/iakovidis_data/Sen1Floods11/splits/flood_handlabeled/flood_bolivia_data.csv"
    train_dataloader = create_dataloader(input_root, label_root, train_csv_name, True, use_vv, batch_size, num_workers)
    valid_dataloader = create_dataloader(input_root, label_root, valid_csv_name, False, use_vv, batch_size//4,
                                         num_workers)
    test_dataloader = create_dataloader(input_root, label_root, test_csv_name, False, use_vv, batch_size // 4,
                                        num_workers)
    bolivia_test_dataloader = create_dataloader(input_root, label_root, bolivia_test_csv_name, False, use_vv,
                                                batch_size // 4, num_workers)
    return train_dataloader, valid_dataloader, test_dataloader, bolivia_test_dataloader


def create_dataloader(input_root, label_root, csv_name, augment, use_vv, batch_size, num_workers):
    data = load_flood_data(input_root, label_root, csv_name, use_vv)
    if augment:
        dataset = InMemoryDataset(data, processAndAugment_unsupervised, use_vv)
        collate_fn = None
    else:
        dataset = InMemoryDataset(data, processTestIm, use_vv)
        collate_fn = lambda x: (torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=None,
                      batch_sampler=None, num_workers=num_workers, collate_fn=collate_fn,
                      pin_memory=True, drop_last=False, timeout=0,
                      worker_init_fn=None)
    return dataloader

train_loader, valid_loader, test_loader, bolivia_test_loader = get_dataloaders(dataset, use_vv, batch_size, num_workers)



# Define the network. For our purposes, we use ResNet50. However, if you wish to test a different model framework,
# optimizer, or loss function you can simply replace those here.
import torch
import torchvision.models as models
import torch.nn as nn

class Unet(nn.Module):

    def __init__(self, input_dim=572, num_input_channels=4, num_output_channels=2, num_first_level_channels=64,
                 depth=5, conv_kernel_size=3, conv_stride=1, max_pool_size=2, up_conv_kernel_size=2, up_conv_stride=2,
                 padding=False, conv_init=None, head=True):
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
        self.conv_init = conv_init
        self.head = head
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
        if self.head:
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

        if self.head:
            x = torch.nn.Softmax(dim=1)(self.final_layer(x))

        return x

    def conv_block(self, num_in_channels, num_out_channels):
        if not self.padding:
            padding_arg = 'valid'
        else:
            padding_arg = 'same'
        conv_layer = nn.Conv2d(num_in_channels, num_out_channels, self.conv_kernel_size, self.conv_stride,
                                             padding=padding_arg)
        if self.conv_init == 'He':
            torch.nn.init.kaiming_uniform_(conv_layer.weight)
        conv_block = nn.Sequential(conv_layer,
                                   nn.BatchNorm2d(num_out_channels),
                                   nn.ReLU())
        return conv_block


    def unet_block(self, num_in_channels, num_out_channels):
        unet_block = nn.Sequential(self.conv_block(num_in_channels, num_out_channels),
                                   self.conv_block(num_out_channels, num_out_channels))
        return unet_block


class Prediction_Module(nn.Module):

    def __init__(self, num_input_channels=4, num_output_channels=2, conv_init=None):
        super().__init__()

        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.conv_init = conv_init
        self.conv_layer = nn.Conv2d(self.num_input_channels, self.num_output_channels, kernel_size=1, stride=1,
                                    padding=0)
        if self.conv_init == "he":
            torch.nn.init.kaiming_uniform_(self.conv_layer.weight)
        self.bn_layer = nn.BatchNorm2d(self.num_output_channels)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn_layer(x)
        return x


input_dim = 256
if use_vv:
    num_input_channels = 2
else:
    num_input_channels = 1
num_output_channels = 2
num_first_level_channels = 64
depth = 4
K = 12

net = Unet(input_dim, num_input_channels, num_output_channels, num_first_level_channels, depth, padding=True,
           conv_init="He", head=False)
net_aug = Unet(input_dim, num_input_channels, num_output_channels, num_first_level_channels, depth, padding=True,
               conv_init="He", head=False)
prediction_module = Prediction_Module(num_first_level_channels, K, conv_init="He")

deep_clustering_loss_func = nn.CrossEntropyLoss()
similarity_loss_func = nn.L1Loss()
optimizer = torch.optim.AdamW(list(net.parameters()) + list(net_aug.parameters()) + list(prediction_module.parameters()),
                              lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_loader) * 10, T_mult=2, eta_min=0,
                                                                 last_epoch=-1)


wandb_config={"learning_rate": lr,
              "batch_size": batch_size,
              "num_workers": num_workers,
              "depth": depth,
              "num_first_level_channels": num_first_level_channels,
              "model": "unet"}


if use_vv:
    wandb_config.update({"training_data": "S1_vv_vh"})
else:
    wandb_config.update({"training_data": "S1_vh"})

if dataset == "flood_hand-labeled":
    wandb_config.update({"dataset":"flood_hand-labeled"})

wandb.init(project="Sen1Floods11-unet", config=wandb_config, tags=['test'])

# Define assessment metrics. For our purposes, we use overall accuracy and mean intersection over union. However, we
# also include functions for calculating true positives, false positives, true negatives, and false negatives.
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

def computeAccuracy(output, target):
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()

    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    correct = torch.sum(output.eq(target))

    return correct.float() / len(target)

def truePositives(output, target):
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()
    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    correct = torch.sum(output * target)

    return correct

def trueNegatives(output, target):
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()
    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    output = (output == 0)
    target = (target == 0)
    correct = torch.sum(output * target)

    return correct

def falsePositives(output, target):
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()
    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    output = (output == 1)
    target = (target == 0)
    correct = torch.sum(output * target)

    return correct

def falseNegatives(output, target):
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()
    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    output = (output == 0)
    target = (target == 1)
    correct = torch.sum(output * target)

    return correct


# Define training loop
training_losses = []
training_accuracies = []
training_ious = []

def train_loop(inputs, inputs_aug, net, net_aug, prediction_module, optimizer, scheduler, epoch_num):
    global running_loss
    global running_deep_clustering_loss
    global running_deep_clustering_loss_aug
    global running_similarity_loss
    global running_disimilarity_loss
    global running_iou
    global running_count
    global running_accuracy

    # zero the parameter gradients
    optimizer.zero_grad()
    net = net.cuda()
    net_aug = net_aug.cuda()
    prediction_module = prediction_module.cuda()

    # forward + backward + optimize
    randomShufflingIndices = torch.randperm(inputs_aug.shape[0])
    inputs_shuffled = inputs_aug[randomShufflingIndices, :, :, :]
    outputs = net(inputs.cuda())
    projections = prediction_module(outputs)
    outputs_aug = net_aug(inputs_aug.cuda())
    projections_aug = prediction_module(outputs_aug)
    outputs_shuffled = net_aug(inputs_shuffled.cuda())
    projections_shuffled = prediction_module(outputs_shuffled)

    _, predictions = torch.max(projections, 1)
    _, predictions_aug = torch.max(projections_aug, 1)
    _, predictions_shuffled = torch.max(projections_shuffled, 1)
    deep_clustering_loss = deep_clustering_loss_func(projections, predictions)
    deep_clustering_loss_aug = deep_clustering_loss_func(projections_aug, predictions_aug)
    similarity_loss = similarity_loss_func(projections, projections_aug)
    disimilarity_loss = -similarity_loss_func(projections, projections_shuffled)
    if epoch_num <50:
        loss = (deep_clustering_loss + deep_clustering_loss_aug)/2
    else:
        loss = (deep_clustering_loss + deep_clustering_loss_aug + similarity_loss + disimilarity_loss) / 4
    loss.backward()
    optimizer.step()
    scheduler.step()

    running_loss += loss.detach().cpu().numpy()
    running_deep_clustering_loss += deep_clustering_loss.detach().cpu().numpy()
    running_deep_clustering_loss_aug += deep_clustering_loss_aug.detach().cpu().numpy()
    running_similarity_loss += similarity_loss.detach().cpu().numpy()
    running_disimilarity_loss += disimilarity_loss.detach().cpu().numpy()
    # running_iou += computeIOU(outputs, labels).detach().numpy()
    # running_accuracy += computeAccuracy(outputs, labels).detach().numpy()
    running_count += 1


# Define validation loop
valid_losses = []
valid_accuracies = []
valid_ious = []

# def validation_loop(validation_data_loader, net, test_data_loader, bolivia_test_loader):
#     global running_loss
#     global running_iou
#     global running_count
#     global running_accuracy
#     global max_valid_iou
#
#     global training_losses
#     global training_accuracies
#     global training_ious
#     global valid_losses
#     global valid_accuracies
#     global valid_ious
#
#     net = net.eval()
#     net = net.cuda()
#     count = 0
#     iou = 0
#     loss = 0
#     accuracy = 0
#     plot_images = True
#     with torch.no_grad():
#         for (images, labels) in validation_data_loader:
#
#             outputs = net(images)
#             valid_loss = criterion(outputs, labels.long())
#             valid_iou = computeIOU(outputs, labels)
#             valid_accuracy = computeAccuracy(outputs, labels)
#             iou += valid_iou.numpy()
#             loss += valid_loss.numpy()
#             accuracy += valid_accuracy.numpy()
#             count += 1
#             if plot_images:
#                 plot_images = False
#                 images = unnorm(images).numpy()
#                 true_segs = labels.numpy().astype(float)
#                 true_segs[true_segs == 255.] = 0.5
#                 model_segs = np.argmax(outputs.detach().numpy(), axis=1)
#                 fig, axs = plt.subplots(4, 10, figsize=(15,8))
#                 cmap = ListedColormap(["lawngreen", "white", "blue"])
#                 for i in range(10):
#                     image = np.transpose(images[i], [1, 2, 0])
#                     true_seg = true_segs[i]
#                     model_seg = model_segs[i]
#                     axs[0, i].imshow(image[:,:,:1], cmap='gray', vmin=0., vmax=1., interpolation='none')
#                     axs[0, i].axis('off')
#                     axs[1, i].imshow(image[:,:,1:], cmap='gray', vmin=0., vmax=1., interpolation='none')
#                     axs[1, i].axis('off')
#                     #                   axs[0, i].set_title('Image')
#                     axs[2, i].imshow(true_seg, cmap=cmap, vmin=0., vmax=1., interpolation='none')
#                     axs[2, i].axis('off')
#                     #                   axs[1, i].set_title('True segmentation map')
#                     axs[3, i].imshow(model_seg, cmap=cmap, vmin=0., vmax=1., interpolation='none')
#                     axs[3, i].axis('off')
#                     #                   axs[2, i].set_title('Predicted segmentation map')
#                 fig.tight_layout()
#                 fig.canvas.draw()
#                 wandb_fig = plt.gcf()
#                 wandb_img = wandb.Image(fig2img(wandb_fig))
#
#     iou = iou / count
#     accuracy = accuracy / count
#
#     loss = loss / count
#     print("Training Loss:", running_loss / running_count)
#     print("Training IOU:", running_iou / running_count)
#     print("Training Accuracy:", running_accuracy / running_count)
#     print("Validation Loss:", loss)
#     print("Validation IOU:", iou)
#     print("Validation Accuracy:", accuracy)
#
#     if iou > max_valid_iou:
#         max_valid_iou = iou
#         save_path = os.path.join("checkpoints", "{}_{}_{}.cp".format(runname, dataset, wandb.run.name))
#         torch.save(net.state_dict(), save_path)
#         print("model saved at", save_path)
#         print("General test dataset:")
#         test_metrics = test_loop(iter(test_data_loader), net)
#         print("Bolivia test dataset:")
#         bolivia_test_metrics = test_loop(iter(bolivia_test_loader), net)
#         metrics = {'train_loss': running_loss / running_count, 'train_iou': running_iou / running_count,
#                    'train_acc': running_accuracy / running_count, 'valid_loss': loss, 'valid_iou': iou,
#                    'valid_acc': accuracy, 'val_example': wandb_img, **test_metrics, **bolivia_test_metrics}
#     else:
#         metrics = {'train_loss': running_loss / running_count, 'train_iou': running_iou / running_count,
#                    'train_acc': running_accuracy / running_count, 'valid_loss': loss, 'valid_iou': iou,
#                    'valid_acc': accuracy, 'val_example': wandb_img}
#
#     training_losses.append(running_loss / running_count)
#     training_accuracies.append(running_accuracy / running_count)
#     training_ious.append(running_iou / running_count)
#     valid_losses.append(loss)
#     valid_accuracies.append(accuracy)
#     valid_ious.append(iou)
#
#     wandb.log(metrics)
#
#
# # Define testing loop (here, you can replace assessment metrics).
# def test_loop(test_data_loader, net):
#     net = net.eval()
#     net = net
#     count = 0
#     iou = 0
#     loss = 0
#     accuracy = 0
#     with torch.no_grad():
#         for (images, labels) in test_data_loader:
#             outputs = net(images)
#             valid_loss = criterion(outputs, labels.long())
#             valid_iou = computeIOU(outputs, labels)
#             iou += valid_iou.numpy()
#             accuracy += computeAccuracy(outputs, labels).numpy()
#             count += 1
#
#     iou = iou / count
#     print("Test IOU:", iou)
#     print("Test Accuracy:", accuracy / count)
#
#     test_metrics = {'test_iou': iou, 'test_acc':accuracy / count}
#
#     return test_metrics


# Define training and validation scheme
running_loss = 0
running_deep_clustering_loss = 0
running_deep_clustering_loss_aug = 0
running_similarity_loss = 0
running_disimilarity_loss = 0
running_iou = 0
running_count = 0
running_accuracy = 0

training_losses = []
training_accuracies = []
training_ious = []
valid_losses = []
valid_accuracies = []
valid_ious = []


def train_epoch(net, net_aug, prediction_module, optimizer, scheduler, train_iter, epoch_num):
    for (inputs, inputs_aug, labels) in train_iter:
        train_loop(inputs, inputs_aug, net, net_aug, prediction_module, optimizer, scheduler, epoch_num)
 

def train_validation_loop(net, net_aug, prediction_module, optimizer, scheduler, train_loader,
                          valid_loader, num_epochs, cur_epoch, test_loader, bolivia_test_loader):
    global running_loss
    global running_deep_clustering_loss
    global running_deep_clustering_loss_aug
    global running_similarity_loss
    global running_disimilarity_loss
    global running_iou
    global running_count
    global running_accuracy
    net = net.train()
    running_loss = 0
    running_deep_clustering_loss = 0
    running_deep_clustering_loss_aug = 0
    running_similarity_loss = 0
    running_disimilarity_loss = 0
    running_iou = 0
    running_count = 0
    running_accuracy = 0

    for i in range(num_epochs):
        train_iter = iter(train_loader)
        train_epoch(net, net_aug, prediction_module, optimizer, scheduler, train_iter, cur_epoch+i+1)

        metrics = {'train_loss': running_loss / running_count,
                   'train_deep_clustering_loss': running_deep_clustering_loss / running_count,
                   'train_deep_clustering_loss_aug': running_deep_clustering_loss_aug / running_count,
                   'train_similarity_loss': running_similarity_loss / running_count,
                   'train_disimilarity_loss': running_disimilarity_loss / running_count}  # , 'clustering_example': wandb_img}

        wandb.log(metrics)

    print("Current Epoch:", cur_epoch)
    # validation_loop(iter(valid_loader), net, net_aug, projector, iter(test_loader), iter(bolivia_test_loader))


# Train model and assess metrics over epochs
import os
import matplotlib.pyplot as plt

max_valid_iou = 0

epochs = []
training_losses = []
training_accuracies = []
training_ious = []
valid_losses = []
valid_accuracies = []
valid_ious = []

for i in range(max_epochs):
    train_validation_loop(net, net_aug, prediction_module, optimizer, scheduler, train_loader, valid_loader, 50, 50*i, test_loader, bolivia_test_loader)
    epochs.append(i)
    x = epochs
    # print("max valid iou:", max_valid_iou)
    if i*50%200 == 0:
        save_path = os.path.join("checkpoints", "unsupervised_{}_{}_{}_{}.cp".format(runname, dataset, wandb.run.name, i*50))
        torch.save(net.state_dict(), save_path)
        save_path = os.path.join("checkpoints",
                                 "unsupervised_aug_{}_{}_{}_{}.cp".format(runname, dataset, wandb.run.name, i * 50))
        torch.save(net_aug.state_dict(), save_path)
        save_path = os.path.join("checkpoints",
                                 "unsupervised_pred_{}_{}_{}_{}.cp".format(runname, dataset, wandb.run.name, i * 50))
        torch.save(prediction_module.state_dict(), save_path)