import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import random
from PIL import Image
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import io


class InMemoryDataset(torch.utils.data.Dataset):
  
  def __init__(self, data_list, preprocess_func):
    self.data_list = data_list
    self.preprocess_func = preprocess_func
  
  def __getitem__(self, i):
    return self.preprocess_func(self.data_list[i])
  
  def __len__(self):
    return len(self.data_list)


def processAndAugment(data):
  (x,y) = data
  im, label = x.copy(), y.copy()

  # convert to PIL for easier transforms
  im2 = Image.fromarray(im[1])
  label = Image.fromarray(label.squeeze())

  # Get params for random transforms
  i, j, h, w = transforms.RandomCrop.get_params(im2, (256, 256))
  
  im2 = F.crop(im2, i, j, h, w)
  label = F.crop(label, i, j, h, w)
  if random.random() > 0.5:
    im2 = F.hflip(im2)
    label = F.hflip(label)
  if random.random() > 0.5:
    im2 = F.vflip(im2)
    label = F.vflip(label)
  
  norm = transforms.Normalize([0.5235], [0.1102])
  im = transforms.ToTensor()(im2)
  im = norm(im)
  label = transforms.ToTensor()(label).squeeze()
  if torch.sum(label.gt(.003) * label.lt(.004)):
    label *= 255
#   label = label.round()

  return im, label


def processTestIm(data):
  (x,y) = data
  im,label = x.copy(), y.copy()
  norm = transforms.Normalize([0.5235], [0.1102])

  # convert to PIL for easier transforms
  im_c2 = Image.fromarray(im[1]).resize((512,512))
  label = Image.fromarray(label.squeeze()).resize((512,512))

  im_c2s = [F.crop(im_c2, 0, 0, 256, 256), F.crop(im_c2, 0, 256, 256, 256),
            F.crop(im_c2, 256, 0, 256, 256), F.crop(im_c2, 256, 256, 256, 256)]
  labels = [F.crop(label, 0, 0, 256, 256), F.crop(label, 0, 256, 256, 256),
            F.crop(label, 256, 0, 256, 256), F.crop(label, 256, 256, 256, 256)]

  ims = [transforms.ToTensor()(x) for x in im_c2s]

  ims = [norm(im) for im in ims]
  ims = torch.stack(ims)

  labels = [(transforms.ToTensor()(label).squeeze()) for label in labels]
  labels = torch.stack(labels)

  if torch.sum(labels.gt(.003) * labels.lt(.004)):
    labels *= 255

  return ims, labels

import csv
import os
import numpy as np
import rasterio

def getArrFlood(fname):
  return rasterio.open(fname).read()

def download_flood_water_data_from_list(l):
  i = 0
  tot_nan = 0
  tot_good = 0
  flood_data = []
  for (im_fname, mask_fname) in l:
    if not os.path.exists(os.path.join("files/", im_fname)):
      continue
    arr_x = np.nan_to_num(getArrFlood(os.path.join("files/", im_fname)))
    arr_y = getArrFlood(os.path.join("files/", mask_fname))
    arr_y[arr_y == -1] = 255

    arr_x = np.clip(arr_x, -50, 1)
    arr_x = (arr_x + 50) / 51
      
    # if i % 100 == 0:
    #   print(im_fname, mask_fname)
    i += 1
    flood_data.append((arr_x,arr_y))

  return flood_data

def load_flood_train_data(input_root, label_root):
  fname = "D:\\Downloads\\flood\\v1.1\splits\\flood_handlabeled\\flood_train_data.csv"
  training_files = []
  with open(fname) as f:
    for line in csv.reader(f):
      training_files.append(tuple((input_root+line[0], label_root+line[1])))

  return download_flood_water_data_from_list(training_files)

def load_flood_valid_data(input_root, label_root):
  fname = "D:\\Downloads\\flood\\v1.1\splits\\flood_handlabeled\\flood_valid_data.csv"
  validation_files = []
  with open(fname) as f:
    for line in csv.reader(f):
      validation_files.append(tuple((input_root+line[0], label_root+line[1])))

  return download_flood_water_data_from_list(validation_files)

def load_flood_test_data(input_root, label_root):
  fname = "D:\\Downloads\\flood\\v1.1\splits\\flood_handlabeled\\flood_test_data.csv"
  testing_files = []
  with open(fname) as f:
    for line in csv.reader(f):
      testing_files.append(tuple((input_root+line[0], label_root+line[1])))
  
  return download_flood_water_data_from_list(testing_files)




train_data = load_flood_train_data("D:\\Downloads\\flood\\v1.1\\data\\flood_events\\HandLabeled\\S1Hand\\",
                                   "D:\\Downloads\\flood\\v1.1\\data\\flood_events\\HandLabeled\\LabelHand\\")
train_dataset = InMemoryDataset(train_data, processAndAugment)

valid_data = load_flood_valid_data("D:\\Downloads\\flood\\v1.1\\data\\flood_events\\HandLabeled\\S1Hand\\",
                                   "D:\\Downloads\\flood\\v1.1\\data\\flood_events\\HandLabeled\\LabelHand\\")
valid_dataset = InMemoryDataset(valid_data, processTestIm)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, sampler=None,
                  batch_sampler=None, num_workers=0, collate_fn=None,
                  pin_memory=True, drop_last=False, timeout=0,
                  worker_init_fn=None)
train_iter = iter(train_loader)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True, sampler=None,
                  batch_sampler=None, num_workers=0, collate_fn=lambda x: (torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0)),
                  pin_memory=True, drop_last=False, timeout=0,
                  worker_init_fn=None)
valid_iter = iter(valid_loader)


import torch.nn as nn

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

input_dim = 256
num_input_channels = 1
num_output_channels = 2
batch_size = 32
num_workers = 8
num_first_level_channels = 64
depth = 4
lr = 0.001

model = Unet(input_dim, num_input_channels, num_output_channels, num_first_level_channels, depth, padding=True)

criterion = nn.CrossEntropyLoss(weight=torch.tensor([1,8]).float(), ignore_index=255)
optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_loader) * 10, T_mult=2, eta_min=0, last_epoch=-1)

wandb.login(key='####')
wandb.init(project="Sen1Floods11-unet", config={
           "learning_rate": lr,
           "batch_size": batch_size,
           "num_workers": num_workers,
           "depth":depth,
           "num_first_level_channels": num_first_level_channels},
           tags=['basic_test'])


# Define assessment metrics. For our purposes, we use overall accuracy and mean intersection over union. However, we also include functions for calculating true positives, false positives, true negatives, and false negatives.
def computeIOU(output, target):
  output = torch.argmax(output, dim=1).flatten() 
  target = target.flatten()
  
  no_ignore = target.ne(255)
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
  
  no_ignore = target.ne(255)
  output = output.masked_select(no_ignore)
  target = target.masked_select(no_ignore)
  correct = torch.sum(output.eq(target))
  
  return correct.float() / len(target)

def truePositives(output, target):
  output = torch.argmax(output, dim=1).flatten() 
  target = target.flatten()
  no_ignore = target.ne(255)
  output = output.masked_select(no_ignore)
  target = target.masked_select(no_ignore)
  correct = torch.sum(output * target)
  
  return correct

def trueNegatives(output, target):
  output = torch.argmax(output, dim=1).flatten() 
  target = target.flatten()
  no_ignore = target.ne(255)
  output = output.masked_select(no_ignore)
  target = target.masked_select(no_ignore)
  output = (output == 0)
  target = (target == 0)
  correct = torch.sum(output * target)
  
  return correct

def falsePositives(output, target):
  output = torch.argmax(output, dim=1).flatten() 
  target = target.flatten()
  no_ignore = target.ne(255)
  output = output.masked_select(no_ignore)
  target = target.masked_select(no_ignore)
  output = (output == 1)
  target = (target == 0)
  correct = torch.sum(output * target)
  
  return correct

def falseNegatives(output, target):
  output = torch.argmax(output, dim=1).flatten() 
  target = target.flatten()
  no_ignore = target.ne(255)
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

def train_loop(inputs, labels, net, optimizer, scheduler):
    global running_loss
    global running_iou
    global running_count
    global running_accuracy

    # zero the parameter gradients
    optimizer.zero_grad()
    net = net

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels.long())
    loss.backward()
    optimizer.step()
    scheduler.step()


    running_loss += loss
    running_iou += computeIOU(outputs, labels)
    running_accuracy += computeAccuracy(outputs, labels)
    running_count += 1
    train_loss = running_loss / running_count
    train_iou = running_iou / running_count
    train_accuracy = running_accuracy / running_count
    print("Training Loss:", train_loss)
    print("Training IOU:", train_iou)
    print("Training Accuracy:", train_accuracy)
    training_losses.append(train_loss.detach().numpy())
    training_accuracies.append(train_accuracy.detach().numpy())
    training_ious.append(train_iou.detach().numpy())
    metrics = {
        'train_loss': train_loss, 'train_iou': train_iou, 'train_acc': train_accuracy
    }
    return metrics


valid_losses = []
valid_accuracies = []
valid_ious = []

def validation_loop(validation_data_loader, net, epoch):
  global running_loss
  global running_iou
  global running_count
  global running_accuracy
  global max_valid_iou

  global training_losses
  global training_accuracies
  global training_ious
  global valid_losses
  global valid_accuracies
  global valid_ious

  net = net.eval()
  net = net
  count = 0
  iou = 0
  loss = 0
  accuracy = 0
  with torch.no_grad():
      for (images, labels) in validation_data_loader:
          net = net
          outputs = net(images)
          valid_loss = criterion(outputs, labels.long())
          valid_iou = computeIOU(outputs, labels)
          valid_accuracy = computeAccuracy(outputs, labels)
          iou += valid_iou
          loss += valid_loss
          accuracy += valid_accuracy
          count += 1

  iou = iou / count
  accuracy = accuracy / count

  if iou > max_valid_iou:
    max_valid_iou = iou
    save_path = os.path.join("checkpoints", "Sen1Floods11_{}_{}.cp".format(epoch, iou.item()))
    torch.save(net.state_dict(), save_path)
    print("model saved at", save_path)

  loss = loss / count
  print("Training Loss:", running_loss / running_count)
  print("Training IOU:", running_iou / running_count)
  print("Training Accuracy:", running_accuracy / running_count)
  print("Validation Loss:", loss)
  print("Validation IOU:", iou)
  print("Validation Accuracy:", accuracy)


  training_losses.append((running_loss / running_count).detach().numpy())
  training_accuracies.append((running_accuracy / running_count).detach().numpy())
  training_ious.append((running_iou / running_count).detach().numpy())
  valid_losses.append(loss.detach().numpy())
  valid_accuracies.append(accuracy.detach().numpy())
  valid_ious.append(iou.detach().numpy())


running_loss = 0
running_iou = 0
running_count = 0
running_accuracy = 0



def train_epoch(net, optimizer, scheduler, train_iter):
  for (inputs, labels) in tqdm(train_iter):
    train_metrics = train_loop(inputs, labels, net, optimizer, scheduler)
  return train_metrics


# Train model and assess metrics over epochs
max_valid_iou = 0

epochs = []
training_losses = []
training_accuracies = []
training_ious = []
valid_losses = []
valid_accuracies = []
valid_ious = []

for epoch in range(1000):
    model = model.train()
    running_loss = 0
    running_iou = 0
    running_count = 0
    running_accuracy = 0
  
    train_iter = iter(train_loader)
    train_metrics = train_epoch(model, optimizer, scheduler, train_iter)
    print("Current Epoch:", epoch)
    val_metrics = validation_loop(iter(valid_loader), model, epoch)
    epochs.append(epoch)
    x = epochs
    plt.plot(x, training_losses, label='training losses')
    plt.plot(x, training_accuracies, 'tab:orange', label='training accuracy')
    plt.plot(x, training_ious, 'tab:purple', label='training iou')
    plt.plot(x, valid_losses, label='valid losses')
    plt.plot(x, valid_accuracies, 'tab:red',label='valid accuracy')
    plt.plot(x, valid_ious, 'tab:green',label='valid iou')
    plt.legend(loc="upper left")
    # plt.show()

    print("max valid iou:", max_valid_iou)
    metrics = {**train_metrics,  **val_metrics}
    wandb.log(metrics)

