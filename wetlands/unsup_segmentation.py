import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F
import random
from PIL import Image
import wandb
import matplotlib.pyplot as plt
import io
import csv
import os
import numpy as np
import rasterio
import cv2

# Define model training hyperparameters
lr = 0.001
max_epochs = 200
batch_size = 16
num_workers = 8
image_size = 512
num_first_level_channels = 8
depth = 2
K = 10
loss_type = "paper"
sim_loss_mult = 1.
disim_loss_mult = 0.1
secondary_losses_epoch = 1
kernel_size = 3
aug_min_s = 1.


# Define functions to process and augment training and testing images
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


# Custom dataset class
class InMemoryDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, preprocess_func, image_size=256):
        self.data_list = data_list
        self.preprocess_func = preprocess_func
        self.image_size = image_size

    def __getitem__(self, i):
        return self.preprocess_func(self.data_list[i], self.image_size)

    def __len__(self):
        return len(self.data_list)


# Function that takes as input the sar image, the label image and the otsu label image, randomly crops and flips them
# and creates an augmented sar image by applying gaussian blurring, and returns the four images.
def processAndAugment_unsupervised(data, image_size=256):
    (x, y, z) = data
    im, label, otsu = x.copy(), y.copy(), z.copy()

    # convert to PIL for easier transforms
    im_vh = Image.fromarray(im[0])
    label = Image.fromarray(label.squeeze())
    otsu = Image.fromarray(otsu.squeeze())

    # Get params for random transforms
    i, j, h, w = transforms.RandomCrop.get_params(im_vh, (image_size, image_size))

    im_vh = F.crop(im_vh, i, j, h, w)
    label = F.crop(label, i, j, h, w)
    otsu = F.crop(otsu, i, j, h, w)
    if random.random() > 0.5:
        im_vh = F.hflip(im_vh)
        label = F.hflip(label)
        otsu = F.hflip(otsu)
    if random.random() > 0.5:
        im_vh = F.vflip(im_vh)
        label = F.vflip(label)
        otsu = F.vflip(otsu)
    im = transforms.ToTensor()(im_vh)
    im_aug = transforms.GaussianBlur(5, sigma=(aug_min_s, 2.0))(im)
    label = transforms.ToTensor()(label).squeeze()
    otsu = transforms.ToTensor()(otsu).squeeze()

    return im, im_aug, label, otsu


# Function that takes as input the sar image, the label image and the otsu label image, resizes them
# and creates an augmented sar image by applying gaussian blurring, and returns the four images.
def processTestIm(data, image_size=256):
    (x, y, z) = data
    im, label, otsu = x.copy(), y.copy(), z.copy()

    # convert to PIL for easier transforms
    im_vhs = Image.fromarray(im[0]).resize((image_size, image_size))
    labels = Image.fromarray(label.squeeze()).resize((image_size, image_size))
    otsu = Image.fromarray(otsu.squeeze()).resize((image_size, image_size))

    im_vhs = [im_vhs]
    labels = [labels]
    otsu = [otsu]

    ims = [transforms.ToTensor()(x) for x in im_vhs]

    ims = torch.stack(ims)
    ims_aug = transforms.GaussianBlur(5, sigma=(aug_min_s, 2.0))(ims)

    labels = [(transforms.ToTensor()(label).squeeze()) for label in labels]
    labels = torch.stack(labels)
    otsu = [(transforms.ToTensor()(o).squeeze()) for o in otsu]
    otsu = torch.stack(otsu)

    return ims, ims_aug, labels, otsu


# Function that reads a .tif file
def getArrFlood(fname):
    return rasterio.open(fname).read()


# Function that takes as input a list of sar and label images and returns a list of tuples each containing a sar image,
# the corresponding label image and the labels produced by the otsu thresholding method
def download_flood_water_data_from_list(l):
    flood_data_x = []
    flood_data_y = []
    for (im_fname, mask_fname) in l:
        if not os.path.exists(im_fname):
            continue
        arr_x = np.nan_to_num(getArrFlood(im_fname))

        arr_y = getArrFlood(mask_fname)

        flood_data_x.append(arr_x)
        flood_data_y.append(arr_y)
    num_images = len(flood_data_x)
    composite_image = (np.concatenate(flood_data_x, axis=1) * 255).astype('uint8')
    blur = cv2.GaussianBlur(composite_image[0], (5, 5), 0)
    threshold, thresholded_image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholded_image[thresholded_image == 255] = 1
    thresholded_image[thresholded_image == 0] = 255
    thresholded_image[thresholded_image == 1] = 0
    otsu = np.split(np.expand_dims(thresholded_image / 255, 0), num_images, 1)
    flood_data = [(flood_data_x[i], flood_data_y[i], otsu[i]) for i in range(num_images)]

    return flood_data


# Function that reads the csv file with the names of the sar and label images and returns the training data in numpy
# arrays
def load_flood_data(input_root, label_root, csv_name):
    files = []
    with open(csv_name, encoding='utf-8') as f:
        for line in csv.reader(f):
            files.append(tuple((input_root + line[0], label_root + line[1])))

    return download_flood_water_data_from_list(files)


# Function that creates the train and validation dataloaders
def get_dataloaders(batch_size, num_workers, image_size=256):
    input_root = "/mimer/NOBACKUP/groups/deep-wetlands-2023/Datasets/Orebro lan_mosaic_2018-07-04_" + str(
        image_size) + "x" + str(image_size) + "_sar/"
    label_root = "/mimer/NOBACKUP/groups/deep-wetlands-2023/Datasets/Orebro lan_mosaic_2018-07-04_" + str(
        image_size) + "x" + str(image_size) + "_ndwi_mask/"
    train_csv_name = "wetlands_" + str(image_size) + "_train_split.csv"
    valid_csv_name = "wetlands_" + str(image_size) + "_test_split.csv"
    train_dataloader = create_dataloader(input_root, label_root, train_csv_name, True, batch_size, num_workers,
                                         image_size)
    valid_dataloader = create_dataloader(input_root, label_root, valid_csv_name, False, batch_size,
                                         num_workers, image_size)
    return train_dataloader, valid_dataloader


# Function that creates a dataloader
def create_dataloader(input_root, label_root, csv_name, augment, batch_size, num_workers, image_size=256):
    data = load_flood_data(input_root, label_root, csv_name)
    if augment:
        dataset = InMemoryDataset(data, processAndAugment_unsupervised, image_size)
        collate_fn = None
    else:
        dataset = InMemoryDataset(data, processTestIm, image_size)
        collate_fn = lambda x: (
        torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0), torch.cat([a[2] for a in x], 0),
        torch.cat([a[3] for a in x], 0))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=None,
                                             batch_sampler=None, num_workers=num_workers, collate_fn=collate_fn,
                                             pin_memory=True, drop_last=False, timeout=0,
                                             worker_init_fn=None)
    return dataloader


train_loader, valid_loader = get_dataloaders(batch_size, num_workers, image_size)
print('datasets created')


# Class for Unet-based projection model
class Unet(nn.Module):

    def __init__(self, input_dim=572, num_first_level_channels=64, depth=5, conv_kernel_size=3, conv_stride=1,
                 max_pool_size=2, up_conv_kernel_size=2, up_conv_stride=2, conv_init=None):
        super().__init__()

        self.input_dim = input_dim
        self.num_input_channels = 1
        self.num_first_level_channels = num_first_level_channels
        self.depth = depth
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.max_pool_size = max_pool_size
        self.up_conv_kernel_size = up_conv_kernel_size
        self.up_conv_stride = up_conv_stride
        self.conv_init = conv_init
        self.depth_num_channels = [self.num_first_level_channels]
        for i in range(self.depth - 1):
            self.depth_num_channels.append(self.depth_num_channels[-1] * 2)
        self.encoder_layers = nn.ModuleList([self.unet_block(self.num_input_channels, self.depth_num_channels[0]),
                                             nn.MaxPool2d(2, return_indices=True)])
        for i in range(len(self.depth_num_channels) - 2):
            self.encoder_layers.append(
                self.unet_block(self.depth_num_channels[i], self.depth_num_channels[i + 1]))
            self.encoder_layers.append(nn.MaxPool2d(2, return_indices=True))
        self.bottleneck = self.unet_block(self.depth_num_channels[-2], self.depth_num_channels[-1])
        self.decoder_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            self.unet_block(self.depth_num_channels[-1] + self.depth_num_channels[-2],
                            self.depth_num_channels[-2])])
        for i in range(len(self.depth_num_channels) - 2, 0, -1):
            self.decoder_layers.append(nn.Upsample(scale_factor=2))
            self.decoder_layers.append(
                self.unet_block(self.depth_num_channels[i] + self.depth_num_channels[i - 1],
                                self.depth_num_channels[i - 1]))

    def forward(self, x):
        skip_connections = []
        for i in range(0, len(self.encoder_layers), 2):
            x = self.encoder_layers[i](x)
            skip_connections.insert(0, x)
            x, max_indices_temp = self.encoder_layers[i + 1](x)

        x = self.bottleneck(x)

        for i in range(0, len(self.decoder_layers), 2):
            x = self.decoder_layers[i](x)
            x = torch.cat([skip_connections[i // 2], x], 1)
            x = self.decoder_layers[i + 1](x)

        return x

    def conv_block(self, num_in_channels, num_out_channels):
        conv_layer = nn.Conv2d(num_in_channels, num_out_channels, self.conv_kernel_size, self.conv_stride,
                               padding='same')
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


# Class for CNN prediction model
class Prediction_Module(nn.Module):

    def __init__(self, num_input_channels=4, num_output_channels=2, conv_init=None):
        super().__init__()

        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.conv_init = conv_init
        self.conv_layer = nn.Conv2d(self.num_input_channels, self.num_output_channels, kernel_size=1, stride=1,
                                    padding=0)
        if self.conv_init == "He":
            torch.nn.init.kaiming_uniform_(self.conv_layer.weight)
        self.bn_layer = nn.BatchNorm2d(self.num_output_channels)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn_layer(x)
        return x


# Create models
net = Unet(image_size, num_first_level_channels, depth, conv_init="He", conv_kernel_size=kernel_size).cuda()
net_aug = Unet(image_size, num_first_level_channels, depth, conv_init="He", conv_kernel_size=kernel_size).cuda()
prediction_module = Prediction_Module(num_first_level_channels, K, conv_init="He").cuda()

# Create loss functions, optimizer and scheduler
deep_clustering_loss_func = nn.CrossEntropyLoss()
similarity_loss_func = nn.L1Loss()
optimizer = torch.optim.AdamW(
    list(net.parameters()) + list(net_aug.parameters()) + list(prediction_module.parameters()),
    lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_loader), T_mult=2, eta_min=0,
                                                                 last_epoch=-1)
# Initialize wandb session
wandb_config = {"learning_rate": lr,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "depth": depth,
                "num_first_level_channels": num_first_level_channels,
                "image_size": image_size,
                "model": "unet",
                "K": K,
                "loss_type": loss_type,
                "secondary_losses_epoch": secondary_losses_epoch,
                "sim_loss_mult": sim_loss_mult,
                "disim_loss_mult": disim_loss_mult,
                'kernel_size': kernel_size,
                'aug_min_s': aug_min_s}

wandb_config.update({"training_data": "wetlands"})

wandb.init(project="wetlands_unsupervised", config=wandb_config)


# Compute and return various different iou metrics.
def computeIOU(output, target):
    batch_size = target.shape[0]
    images_with_wetlands = torch.max(torch.max(target == 1., dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
    num_nd_images = torch.sum(images_with_wetlands)
    indices_keep = images_with_wetlands.expand(batch_size, image_size, image_size)
    indices_keep = indices_keep.flatten()
    perc_wetlands = torch.sum(target == 1, dim=[1, 2]) / (image_size * image_size)
    im_intersection = output * target
    im_union = target + output - im_intersection
    im_iou = (torch.sum(im_intersection, dim=[1, 2]) + .0000001) / (torch.sum(im_union, dim=[1, 2]) + .0000001)
    dry_image_mask = perc_wetlands <= 0.1
    wet_image_mask = perc_wetlands >= 0.9
    rest_image_mask = (0.1 < perc_wetlands) & (perc_wetlands < 0.9)
    mean_im_iou = torch.mean(im_iou)
    num_im = im_iou.shape[0]
    mean_dry_im_iou = torch.mean(im_iou.masked_select(dry_image_mask))
    num_dry_im = torch.sum(dry_image_mask)
    mean_wet_im_iou = torch.mean(im_iou.masked_select(wet_image_mask))
    num_wet_im = torch.sum(wet_image_mask)
    mean_rest_im_iou = torch.mean(im_iou.masked_select(rest_image_mask))
    num_rest_im = torch.sum(rest_image_mask)
    output = output.flatten()
    target = target.flatten()
    intersection = output * target
    union = target + output - intersection
    iou = (torch.sum(intersection) + .0000001) / (torch.sum(union) + .0000001)
    dry_iou = (torch.sum(
        intersection.masked_select(dry_image_mask.repeat_interleave(image_size * image_size))) + .0000001) / (torch.sum(
        union.masked_select(dry_image_mask.repeat_interleave(image_size * image_size))) + .0000001)
    wet_iou = (torch.sum(
        intersection.masked_select(wet_image_mask.repeat_interleave(image_size * image_size))) + .0000001) / (
                      torch.sum(union.masked_select(
                          wet_image_mask.repeat_interleave(image_size * image_size))) + .0000001)
    rest_iou = (torch.sum(
        intersection.masked_select(rest_image_mask.repeat_interleave(image_size * image_size))) + .0000001) / (
                       torch.sum(union.masked_select(
                           rest_image_mask.repeat_interleave(image_size * image_size))) + .0000001)
    output_nd = output.masked_select(indices_keep)
    target_nd = target.masked_select(indices_keep)
    intersection_nd = torch.sum(output_nd * target_nd)
    union_nd = torch.sum(target_nd) + torch.sum(output_nd) - intersection_nd
    iou_nd = (intersection_nd + .0000001) / (union_nd + .0000001)
    return mean_im_iou.cpu().numpy(), num_im, mean_dry_im_iou.cpu().numpy(), num_dry_im.cpu().numpy(), mean_wet_im_iou.cpu().numpy(), \
           num_wet_im.cpu().numpy(), mean_rest_im_iou.cpu().numpy(), num_rest_im.cpu().numpy(), iou.cpu().numpy(), dry_iou.cpu().numpy(), \
           wet_iou.cpu().numpy(), rest_iou.cpu().numpy(), iou_nd.cpu().numpy(), num_nd_images.cpu().numpy(), im_iou.numpy()


# Compute per-pixel accuracy
def computeAccuracy(output, target):
    output = output.flatten()
    target = target.flatten()

    correct = torch.sum(output.eq(target))

    return correct.float() / len(target)


# Mach predicted classes to water/land by assigning each class to the area that is has the greatest iou with.
def matchSegmentationResultToOriginalLabel(resultMaps, referenceMaps):
    referenceMaps = referenceMaps.astype(int)
    original_shape = resultMaps.shape
    resultMaps = resultMaps.flatten()
    referenceMaps = referenceMaps.flatten()
    ##ADDING 1 to not keep any zero value
    ##Otherwise zero is an object here (Impervious surfaces)
    resultMaps = resultMaps + 1
    referenceMaps = referenceMaps + 1

    ##Finding unique values
    resultMapUniqueVals = np.unique(resultMaps)

    referenceMapUniqueVals, referenceMapUniqueCounts = np.unique(referenceMaps, return_counts=True)
    referenceSortingIndices = np.argsort(-referenceMapUniqueCounts)
    referenceMapUniqueVals = referenceMapUniqueVals[referenceSortingIndices]

    resultToReferenceRelationMatrix = np.zeros((len(resultMapUniqueVals), len(referenceMapUniqueVals)))
    for resultIndex, resultUniqueVal in enumerate(resultMapUniqueVals):
        resultUniqueValIndicator = np.copy(resultMaps)
        resultUniqueValIndicator[resultUniqueValIndicator != resultUniqueVal] = 0
        for referenceIndex, referenceUniqueVal in enumerate(referenceMapUniqueVals):
            referenceUniqueValIndicator = np.copy(referenceMaps)
            referenceUniqueValIndicator[referenceUniqueValIndicator != referenceUniqueVal] = 0
            resultReferenceIntersection = resultUniqueValIndicator * referenceUniqueValIndicator
            numIntersection = len(np.argwhere(resultReferenceIntersection))
            num_union = len(np.argwhere(resultUniqueValIndicator + referenceUniqueValIndicator))
            resultToReferenceRelationMatrix[resultIndex, referenceIndex] = numIntersection / num_union

    resultMapReassigned = np.zeros(resultMaps.shape)

    for resultIndex, resultUniqueVal in enumerate(resultMapUniqueVals):
        matchesCorrespondingToThisVal = resultToReferenceRelationMatrix[resultIndex, :]
        maximizingIndex = np.argsort(matchesCorrespondingToThisVal)[-1]
        resultMapOptimumMatch = referenceMapUniqueVals[maximizingIndex]
        resultMapReassigned[resultMaps == resultUniqueVal] = resultMapOptimumMatch

    ##Subtracting 1 to keep values as it were
    resultMapReassigned = resultMapReassigned - 1

    resultMapReassigned = np.reshape(resultMapReassigned, original_shape).astype(int)

    return torch.from_numpy(resultMapReassigned)


# Function that performs one epoch of validation
def validation_loop(net, net_aug, prediction_module, validation_data_loader, epoch):
    global max_valid_iou
    global max_otsu_valid_iou

    valid_loss = 0
    valid_deep_clustering_loss = 0
    valid_deep_clustering_loss_aug = 0
    valid_similarity_loss = 0
    valid_disimilarity_loss = 0

    num_im_plot = 8
    fig, axs = plt.subplots(5, num_im_plot, figsize=(2 * num_im_plot, 9))
    ts_palette = np.array([[0, 255, 0],  # green
                           [0, 0, 255]])  # blue
    model_palette = np.array([[0, 0, 0],
                              [255, 0, 0],
                              [0, 255, 0],
                              [0, 0, 255],
                              [255, 255, 255],
                              [255, 255, 0],
                              [255, 0, 255],
                              [0, 255, 255],
                              [127, 127, 127],
                              [0, 127, 127],
                              [127, 0, 127],
                              [127, 127, 0],
                              [0, 0, 127],
                              [0, 127, 0],
                              [127, 0, 0]])

    num_im = 0
    num_dry_im = 0
    num_wet_im = 0
    num_rest_im = 0
    num_nd_im = 0
    iou = 0
    nd_iou = 0
    dry_iou = 0
    wet_iou = 0
    rest_iou = 0
    im_iou = 0
    im_dry_iou = 0
    im_wet_iou = 0
    im_rest_iou = 0
    accuracy = 0
    otsu_iou = 0
    otsu_nd_iou = 0
    otsu_dry_iou = 0
    otsu_wet_iou = 0
    otsu_rest_iou = 0
    otsu_im_iou = 0
    otsu_im_dry_iou = 0
    otsu_im_wet_iou = 0
    otsu_im_rest_iou = 0
    otsu_accuracy = 0
    count = 0
    num_batches = 0

    val_predictions = []
    val_labels = []
    val_images = []
    val_otsu = []
    with torch.no_grad():
        for (inputs, inputs_aug, labels, otsu) in validation_data_loader:
            inputs = inputs.cuda()
            inputs_aug = inputs_aug.cuda()
            labels = labels.cuda()
            otsu = otsu.cuda()
            batch_num_im = inputs.shape[0]

            randomShufflingIndices = torch.randperm(inputs_aug.shape[0])
            original_indices = torch.range(0, inputs_aug.shape[0] - 1)
            while torch.sum(torch.eq(randomShufflingIndices, original_indices)) > 0:
                randomShufflingIndices = torch.randperm(inputs_aug.shape[0])

            inputs_shuffled = inputs_aug[randomShufflingIndices, :, :, :]
            outputs = net(inputs)
            projections = prediction_module(outputs)
            outputs_aug = net_aug(inputs_aug)
            projections_aug = prediction_module(outputs_aug)
            outputs_shuffled = net_aug(inputs_shuffled)
            projections_shuffled = prediction_module(outputs_shuffled)

            _, predictions = torch.max(projections, 1)
            _, predictions_aug = torch.max(projections_aug, 1)
            _, predictions_shuffled = torch.max(projections_shuffled, 1)
            deep_clustering_loss = deep_clustering_loss_func(projections, predictions)
            deep_clustering_loss_aug = deep_clustering_loss_func(projections_aug, predictions_aug)
            similarity_loss = similarity_loss_func(projections, projections_aug)
            disimilarity_loss = -similarity_loss_func(projections, projections_shuffled)
            if loss_type == "paper":
                if epoch < secondary_losses_epoch:
                    loss = (deep_clustering_loss + deep_clustering_loss_aug) / 2
                else:
                    loss = (deep_clustering_loss + deep_clustering_loss_aug + sim_loss_mult * similarity_loss +
                            disim_loss_mult * disimilarity_loss) / (2 + sim_loss_mult + disim_loss_mult)
            elif loss_type == "reverse_paper":
                if epoch >= secondary_losses_epoch:
                    loss = (deep_clustering_loss + deep_clustering_loss_aug) / 2
                else:
                    loss = (deep_clustering_loss + deep_clustering_loss_aug + sim_loss_mult * similarity_loss +
                            disim_loss_mult * disimilarity_loss) / (2 + sim_loss_mult + disim_loss_mult)
            valid_loss += loss.detach().cpu().numpy() * batch_num_im
            valid_deep_clustering_loss += deep_clustering_loss.detach().cpu().numpy() * batch_num_im
            valid_deep_clustering_loss_aug += deep_clustering_loss_aug.detach().cpu().numpy() * batch_num_im
            valid_similarity_loss += similarity_loss.detach().cpu().numpy() * batch_num_im
            valid_disimilarity_loss += disimilarity_loss.detach().cpu().numpy() * batch_num_im

            val_predictions.append(predictions.cpu().numpy())
            val_labels.append(labels.cpu().numpy())
            val_images.append(inputs.cpu().numpy())
            val_otsu.append(otsu.cpu().numpy())
            num_batches += 1
    val_predictions = np.concatenate(val_predictions, axis=0)
    val_images = np.concatenate(val_images, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    val_otsu = np.concatenate(val_otsu, axis=0)
    matched_model_segs = matchSegmentationResultToOriginalLabel(val_predictions, val_labels)
    for j in range(num_batches):
        images = val_images[j * batch_size:min(val_images.shape[0], (j + 1) * batch_size)]
        labels = val_labels[j * batch_size:min(val_labels.shape[0], (j + 1) * batch_size)]
        otsu = val_otsu[j * batch_size:min(val_otsu.shape[0], (j + 1) * batch_size)]
        predictions = val_predictions[j * batch_size:min(val_predictions.shape[0], (j + 1) * batch_size)]
        matched_model_seg = matched_model_segs[j * batch_size:min(matched_model_segs.shape[0], (j + 1) * batch_size)]
        batch_mean_im_iou, batch_num_im, batch_mean_dry_im_iou, batch_num_dry_im, batch_mean_wet_im_iou, batch_num_wet_im, \
        batch_mean_rest_im_iou, batch_num_rest_im, batch_iou, batch_dry_iou, batch_wet_iou, batch_rest_iou, batch_nd_iou, batch_num_nd_im, sep_im_iou = \
            computeIOU(matched_model_seg, torch.from_numpy(labels))
        valid_accuracy = computeAccuracy(matched_model_seg, torch.from_numpy(labels))
        iou += batch_iou * batch_num_im
        dry_iou += np.nan_to_num(batch_dry_iou) * batch_num_dry_im
        wet_iou += np.nan_to_num(batch_wet_iou) * batch_num_wet_im
        rest_iou += np.nan_to_num(batch_rest_iou) * batch_num_rest_im
        im_iou += batch_mean_im_iou * batch_num_im
        im_dry_iou += np.nan_to_num(batch_mean_dry_im_iou) * batch_num_dry_im
        im_wet_iou += np.nan_to_num(batch_mean_wet_im_iou) * batch_num_wet_im
        im_rest_iou += np.nan_to_num(batch_mean_rest_im_iou) * batch_num_rest_im
        num_im += batch_num_im
        num_dry_im += batch_num_dry_im
        num_wet_im += batch_num_wet_im
        num_rest_im += batch_num_rest_im
        nd_iou += np.nan_to_num(batch_nd_iou) * batch_num_nd_im
        num_nd_im += batch_num_nd_im
        accuracy += valid_accuracy.numpy() * batch_num_im

        otsu_batch_mean_im_iou, _, otsu_batch_mean_dry_im_iou, _, otsu_batch_mean_wet_im_iou, _, otsu_batch_mean_rest_im_iou, \
        _, otsu_batch_iou, otsu_batch_dry_iou, otsu_batch_wet_iou, otsu_batch_rest_iou, otsu_batch_nd_iou, _, otsu_sep_im_iou = \
            computeIOU(torch.from_numpy(otsu), torch.from_numpy(labels))
        otsu_valid_accuracy = computeAccuracy(torch.from_numpy(otsu), torch.from_numpy(labels))
        otsu_iou += otsu_batch_iou * batch_num_im
        otsu_dry_iou += np.nan_to_num(otsu_batch_dry_iou) * batch_num_dry_im
        otsu_wet_iou += np.nan_to_num(otsu_batch_wet_iou) * batch_num_wet_im
        otsu_rest_iou += np.nan_to_num(otsu_batch_rest_iou) * batch_num_rest_im
        otsu_im_iou += otsu_batch_mean_im_iou * batch_num_im
        otsu_im_dry_iou += np.nan_to_num(otsu_batch_mean_dry_im_iou) * batch_num_dry_im
        otsu_im_wet_iou += np.nan_to_num(otsu_batch_mean_wet_im_iou) * batch_num_wet_im
        otsu_im_rest_iou += np.nan_to_num(otsu_batch_mean_rest_im_iou) * batch_num_rest_im
        otsu_accuracy += otsu_valid_accuracy.numpy() * batch_num_im
        otsu_nd_iou += np.nan_to_num(otsu_batch_nd_iou) * batch_num_nd_im

        if num_im_plot > 0:
            true_segs = labels.astype(int)
            model_segs = predictions.astype(int)
            matched_segs = matched_model_seg.numpy().astype(int)
            otsu_segs = otsu.astype(int)
            for i in range(batch_num_im):
                num_im_plot -= 1
                image = np.transpose(images[i], [1, 2, 0])
                image_iou = sep_im_iou[i]
                otsu_image_iou = otsu_sep_im_iou[i]
                true_seg = ts_palette[true_segs[i]]
                model_seg = model_palette[model_segs[i]]
                matched_seg = ts_palette[matched_segs[i]]
                otsu_seg = ts_palette[otsu_segs[i]]
                axs[0, num_im_plot].imshow(image[:, :, :1], cmap='gray', vmin=0., vmax=1., interpolation='none')
                axs[1, num_im_plot].imshow(true_seg, interpolation='none')
                axs[2, num_im_plot].imshow(model_seg, interpolation='none')
                axs[3, num_im_plot].imshow(matched_seg, interpolation='none')
                axs[3, num_im_plot].set_title('Iou: ' + str(image_iou.round(2)))
                axs[4, num_im_plot].imshow(otsu_seg, interpolation='none')
                axs[4, num_im_plot].set_title('Iou: ' + str(otsu_image_iou.round(2)))
                if num_im_plot == 0:
                    axs[0, 0].set(ylabel='SAR Image')
                    axs[1, 0].set(ylabel='Ground Truth')
                    axs[2, 0].set(ylabel='Model Classes')
                    axs[3, 0].set(ylabel='Model Labels')
                    axs[4, 0].set(ylabel='Otsu Labels')
                    plt.setp(axs, xticks=[], yticks=[])
                    fig.tight_layout()
                    fig.canvas.draw()
                    wandb_fig = plt.gcf()
                    wandb_img = wandb.Image(fig2img(wandb_fig))
                    break
    nd_iou = nd_iou / num_nd_im
    accuracy = accuracy / num_im
    iou = iou / num_im
    dry_iou = dry_iou / num_dry_im
    wet_iou = wet_iou / num_wet_im
    rest_iou = rest_iou / num_rest_im
    im_iou = im_iou / num_im
    im_dry_iou = im_dry_iou / num_dry_im
    im_wet_iou = im_wet_iou / num_wet_im
    im_rest_iou = im_rest_iou / num_rest_im

    otsu_nd_iou = otsu_nd_iou / num_nd_im
    otsu_accuracy = otsu_accuracy / num_im
    otsu_iou = otsu_iou / num_im
    otsu_dry_iou = otsu_dry_iou / num_dry_im
    otsu_wet_iou = otsu_wet_iou / num_wet_im
    otsu_rest_iou = otsu_rest_iou / num_rest_im
    otsu_im_iou = otsu_im_iou / num_im
    otsu_im_dry_iou = otsu_im_dry_iou / num_dry_im
    otsu_im_wet_iou = otsu_im_wet_iou / num_wet_im
    otsu_im_rest_iou = otsu_im_rest_iou / num_rest_im

    valid_loss = valid_loss / num_im
    valid_deep_clustering_loss = valid_deep_clustering_loss / num_im
    valid_deep_clustering_loss_aug = valid_deep_clustering_loss_aug / num_im
    valid_similarity_loss = valid_similarity_loss / num_im
    valid_disimilarity_loss = valid_disimilarity_loss / num_im

    if iou > max_valid_iou:
        max_valid_iou = iou
    if otsu_iou > max_otsu_valid_iou:
        max_otsu_valid_iou = otsu_iou

    print("Validation IOU:", iou)
    print("Validation Accuracy:", accuracy)

    valid_metrics = {'valid_iou': iou, 'valid_dry_iou': dry_iou, 'valid_wet_iou': wet_iou, 'valid_rest_iou': rest_iou,
                     'valid_im_iou': im_iou, 'valid_im_dry_iou': im_dry_iou, 'valid_im_wet_iou': im_wet_iou,
                     'valid_im_rest_iou': im_rest_iou, 'valid_nd_iou': nd_iou, 'valid_acc': accuracy,
                     'val_example': wandb_img,
                     'valid_loss': valid_loss, 'valid_deep_clustering_loss': valid_deep_clustering_loss,
                     'valid_deep_clustering_loss_aug': valid_deep_clustering_loss_aug,
                     'valid_similarity_loss': valid_similarity_loss, 'valid_disimilarity_loss': valid_disimilarity_loss,
                     'max_valid_iou': max_valid_iou, 'otsu_valid_iou': otsu_iou, 'otsu_valid_dry_iou': otsu_dry_iou,
                     'otsu_valid_wet_iou': otsu_wet_iou,
                     'otsu_valid_rest_iou': otsu_rest_iou, 'otsu_valid_im_iou': otsu_im_iou,
                     'otsu_valid_im_dry_iou': otsu_im_dry_iou,
                     'otsu_valid_im_wet_iou': otsu_im_wet_iou, 'otsu_valid_im_rest_iou': otsu_im_rest_iou,
                     'otsu_valid_acc': otsu_accuracy,
                     'max_otsu_valid_iou': max_otsu_valid_iou, 'otsu_valid_nd_iou': otsu_nd_iou}

    return valid_metrics


# Function that performs one epoch of training
def train_epoch(net, net_aug, prediction_module, optimizer, scheduler, train_iter, epoch_num):
    train_loss = 0
    train_deep_clustering_loss = 0
    train_deep_clustering_loss_aug = 0
    train_similarity_loss = 0
    train_disimilarity_loss = 0

    num_im = 0
    num_dry_im = 0
    num_wet_im = 0
    num_rest_im = 0
    num_nd_im = 0
    iou = 0
    nd_iou = 0
    dry_iou = 0
    wet_iou = 0
    rest_iou = 0
    im_iou = 0
    im_dry_iou = 0
    im_wet_iou = 0
    im_rest_iou = 0
    accuracy = 0
    num_batches = 0

    train_predictions = []
    train_labels = []

    for (inputs, inputs_aug, labels, otsu) in train_iter:
        inputs = inputs.cuda()
        inputs_aug = inputs_aug.cuda()
        labels = labels.cuda()
        batch_num_im = inputs.shape[0]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        randomShufflingIndices = torch.randperm(inputs_aug.shape[0])
        original_indices = torch.range(0, inputs_aug.shape[0] - 1)
        while torch.sum(torch.eq(randomShufflingIndices, original_indices)) > 0:
            randomShufflingIndices = torch.randperm(inputs_aug.shape[0])
        inputs_shuffled = inputs_aug[randomShufflingIndices, :, :, :]
        outputs = net(inputs)
        projections = prediction_module(outputs)
        outputs_aug = net_aug(inputs_aug)
        projections_aug = prediction_module(outputs_aug)
        outputs_shuffled = net_aug(inputs_shuffled)
        projections_shuffled = prediction_module(outputs_shuffled)

        _, predictions = torch.max(projections, 1)
        values, counts = torch.unique(predictions, sorted=True, return_counts=True)
        _, predictions_aug = torch.max(projections_aug, 1)
        values_aug, counts_aug = torch.unique(predictions_aug, sorted=True, return_counts=True)
        _, predictions_shuffled = torch.max(projections_shuffled, 1)
        weights = torch.zeros((K,), device='cuda')
        class_counts = torch.ones((K,), device='cuda')
        for i in range(values.shape[0]):
            class_counts[values[i]] += counts[i]
        for i in range(values_aug.shape[0]):
            class_counts[values_aug[i]] += counts_aug[i]
        for i in range(K):
            weights[i] = 0.00001 / (class_counts[i] + 0.00001)
        weights = weights / torch.sum(weights)
        deep_clustering_loss_func = nn.CrossEntropyLoss(weight=weights)

        deep_clustering_loss = deep_clustering_loss_func(projections, predictions)
        deep_clustering_loss_aug = deep_clustering_loss_func(projections_aug, predictions_aug)
        similarity_loss = similarity_loss_func(projections, projections_aug)
        disimilarity_loss = -similarity_loss_func(projections, projections_shuffled)
        if loss_type == "paper":
            if epoch_num < secondary_losses_epoch:
                loss = (deep_clustering_loss + deep_clustering_loss_aug) / 2
            else:
                loss = (deep_clustering_loss + deep_clustering_loss_aug + sim_loss_mult * similarity_loss +
                        disim_loss_mult * disimilarity_loss) / (2 + sim_loss_mult + disim_loss_mult)
        elif loss_type == "reverse_paper":
            if epoch_num >= secondary_losses_epoch:
                loss = (deep_clustering_loss + deep_clustering_loss_aug) / 2
            else:
                loss = (deep_clustering_loss + deep_clustering_loss_aug + sim_loss_mult * similarity_loss +
                        disim_loss_mult * disimilarity_loss) / (2 + sim_loss_mult + disim_loss_mult)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.detach().cpu().numpy() * batch_num_im
        train_deep_clustering_loss += deep_clustering_loss.detach().cpu().numpy() * batch_num_im
        train_deep_clustering_loss_aug += deep_clustering_loss_aug.detach().cpu().numpy() * batch_num_im
        train_similarity_loss += similarity_loss.detach().cpu().numpy() * batch_num_im
        train_disimilarity_loss += disimilarity_loss.detach().cpu().numpy() * batch_num_im

        train_predictions.append(predictions.cpu().numpy())
        train_labels.append(labels.cpu().numpy())
        # train_images.append(inputs.cpu().numpy())
        num_batches += 1

    train_predictions = np.concatenate(train_predictions, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    matched_model_segs = matchSegmentationResultToOriginalLabel(train_predictions, train_labels)

    for j in range(num_batches):
        labels = train_labels[j * batch_size:min(train_labels.shape[0], (j + 1) * batch_size)]
        matched_model_seg = matched_model_segs[j * batch_size:min(matched_model_segs.shape[0], (j + 1) * batch_size)]
        batch_mean_im_iou, batch_num_im, batch_mean_dry_im_iou, batch_num_dry_im, batch_mean_wet_im_iou, batch_num_wet_im, \
        batch_mean_rest_im_iou, batch_num_rest_im, batch_iou, batch_dry_iou, batch_wet_iou, batch_rest_iou, batch_nd_iou, batch_num_nd_im, sep_im_iou = \
            computeIOU(matched_model_seg, torch.from_numpy(labels))
        train_accuracy = computeAccuracy(matched_model_seg, torch.from_numpy(labels))
        iou += batch_iou * batch_num_im
        dry_iou += np.nan_to_num(batch_dry_iou) * batch_num_dry_im
        wet_iou += np.nan_to_num(batch_wet_iou) * batch_num_wet_im
        rest_iou += np.nan_to_num(batch_rest_iou) * batch_num_rest_im
        im_iou += batch_mean_im_iou * batch_num_im
        im_dry_iou += np.nan_to_num(batch_mean_dry_im_iou) * batch_num_dry_im
        im_wet_iou += np.nan_to_num(batch_mean_wet_im_iou) * batch_num_wet_im
        im_rest_iou += np.nan_to_num(batch_mean_rest_im_iou) * batch_num_rest_im
        num_im += batch_num_im
        num_dry_im += batch_num_dry_im
        num_wet_im += batch_num_wet_im
        num_rest_im += batch_num_rest_im
        nd_iou += np.nan_to_num(batch_nd_iou) * batch_num_nd_im
        num_nd_im += batch_num_nd_im

        accuracy += train_accuracy.numpy() * batch_num_im

    nd_iou = nd_iou / num_nd_im
    accuracy = accuracy / num_im
    iou = iou / num_im
    dry_iou = dry_iou / num_dry_im
    wet_iou = wet_iou / num_wet_im
    rest_iou = rest_iou / num_rest_im
    im_iou = im_iou / num_im
    im_dry_iou = im_dry_iou / num_dry_im
    im_wet_iou = im_wet_iou / num_wet_im
    im_rest_iou = im_rest_iou / num_rest_im

    train_loss = train_loss / num_im
    train_deep_clustering_loss = train_deep_clustering_loss / num_im
    train_deep_clustering_loss_aug = train_deep_clustering_loss_aug / num_im
    train_similarity_loss = train_similarity_loss / num_im
    train_disimilarity_loss = train_disimilarity_loss / num_im

    train_metrics = {'train_iou': iou, 'train_dry_iou': dry_iou, 'train_wet_iou': wet_iou, 'train_rest_iou': rest_iou,
                     'train_im_iou': im_iou, 'train_im_dry_iou': im_dry_iou, 'train_im_wet_iou': im_wet_iou,
                     'train_im_rest_iou': im_rest_iou, 'train_nd_iou': nd_iou, 'train_acc': accuracy,
                     'train_loss': train_loss, 'train_deep_clustering_loss': train_deep_clustering_loss,
                     'train_deep_clustering_loss_aug': train_deep_clustering_loss_aug,
                     'train_similarity_loss': train_similarity_loss, 'train_disimilarity_loss': train_disimilarity_loss}
    return train_metrics


# Function that performs training and validation
def train_validation_loop(net, net_aug, prediction_module, optimizer, scheduler, train_loader, valid_loader, cur_epoch):
    net = net.train()
    net_aug = net_aug.train()
    prediction_module = prediction_module.train()

    train_metrics = train_epoch(net, net_aug, prediction_module, optimizer, scheduler, iter(train_loader), cur_epoch)

    net = net.eval()
    net_aug = net_aug.eval()
    prediction_module = prediction_module.eval()

    valid_metrics = validation_loop(net, net_aug, prediction_module, iter(valid_loader), cur_epoch)

    metrics = {**train_metrics, **valid_metrics}
    wandb.log(metrics)


# Train model and assess metrics over epochs
max_valid_iou = 0
max_otsu_valid_iou = 0

for i in range(max_epochs):
    train_validation_loop(net, net_aug, prediction_module, optimizer, scheduler, train_loader, valid_loader, i)
