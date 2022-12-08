from gsv_scraper import IMAGES_CSV, IMAGES_DIR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import io
import torchvision.transforms as T
import numpy as np
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter

RESOLUTION = 30
RESIZE_WIDTH = 240
RESIZE_HEIGHT = 180
NUM_EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATES = [0.001, 0.0005, 0.0001]
MOMENTUM = 0.9

TRAIN_OUT = 'train_out'
MODEL_PATH = os.path.join(TRAIN_OUT, 'model.pt')
LOSSES_CSV = os.path.join(TRAIN_OUT, 'losses.csv')
ACCURACIES_CSV = os.path.join(TRAIN_OUT, 'accuracies.csv')
LOSSES_PLOT = os.path.join(TRAIN_OUT, 'losses.png')
ACCURACIES_PLOT = os.path.join(TRAIN_OUT, 'accuracies.png')

start_time = perf_counter()

class GeoData(Dataset):
  def __init__(self, images_df):
    super(Dataset, self).__init__()
    self.images_df = images_df
    self.resize = T.Resize((RESIZE_HEIGHT, RESIZE_WIDTH))

  def __len__(self):
    return len(self.images_df)

  def __getitem__(self, i):
    image = self.images_df.iloc[i]
    pano_id = image['pano_id']
    lat = image['lat']
    lng = image['lng']

    image_path = os.path.join(IMAGES_DIR, f'{pano_id}.png')
    image = io.read_image(image_path).float()
    if image.shape[0] == 4:  # remove alpha channel
      image = image[:3]
    image = self.resize(image)

    label = torch.LongTensor([(360 // RESOLUTION) * ((lat + 90) // RESOLUTION) + ((lng + 180) // RESOLUTION)])

    return image, label

class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ResBlock, self).__init__()
    self.downsample = out_channels > in_channels
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 if self.downsample else 1, padding=1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    if self.downsample:
      self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
      self.bn3 = nn.BatchNorm2d(out_channels)

  def forward(self, x):
    r = x
    x = self.conv1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    if self.downsample:
      r = self.conv3(r)
      r = self.bn3(r)
    x += r
    x = F.relu(x)
    return x

class GeoNet(nn.Module):
  def __init__(self):
    super(GeoNet, self).__init__()
    self.conv = nn.Conv2d(3, 8, kernel_size=7, stride=2, padding=3)
    self.bn = nn.BatchNorm2d(8)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.res_blocks = [
      ResBlock(8, 8),
      ResBlock(8, 8),
      ResBlock(8, 16),
      ResBlock(16, 16),
      ResBlock(16, 32),
      ResBlock(32, 32),
      ResBlock(32, 64),
      ResBlock(64, 64),
      ResBlock(64, 128),
      ResBlock(128, 128)
    ]
    self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    self.fc = nn.Linear(128, (180 // RESOLUTION) * (360 // RESOLUTION))

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = F.relu(x)
    x = self.maxpool(x)
    for res_block in self.res_blocks:
      x = res_block(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x

  def loss(self, pred, labels):
    return F.cross_entropy(pred, labels)

def train(model, train_loader, optimizer):
  model.train()
  losses = []
  accuracies = []
  for batch, (images, labels) in enumerate(train_loader):
    labels = labels.squeeze()
    optimizer.zero_grad()
    pred = model(images)
    loss = model.loss(pred, labels)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    accuracy = torch.sum(torch.argmax(pred, 1) == labels) / len(labels)
    accuracies.append(accuracy.item())
    print(f'batch: {batch + 1}/{len(train_loader)}, train loss: {loss.item()}, train accuracy: {accuracy.item()}, time: {round(perf_counter() - start_time, 1)}s')
  train_loss = np.mean(losses)
  train_accuracy = np.mean(accuracies)
  return train_loss, train_accuracy

def test(model, test_loader):
  print('testing...')
  model.eval()
  losses = []
  accuracies = []
  with torch.no_grad():
    for batch, (images, labels) in enumerate(test_loader):
      labels = labels.squeeze()
      pred = model(images)
      loss = model.loss(pred, labels)
      losses.append(loss.item())
      accuracy = torch.sum(torch.argmax(pred, 1) == labels) / len(labels)
      accuracies.append(accuracy.item())
  test_loss = np.mean(losses)
  test_accuracy = np.mean(accuracies)
  return test_loss, test_accuracy

def main():
  images_df = pd.read_csv(IMAGES_CSV)
  train_size = int(0.9 * len(images_df))
  train_data = GeoData(images_df[:train_size])
  test_data = GeoData(images_df[train_size:])
  train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

  if not os.path.isdir(TRAIN_OUT):
    os.makedirs(TRAIN_OUT)
    with open(LOSSES_CSV, 'w') as losses_csv:
      loss_writer = csv.writer(losses_csv)
      loss_writer.writerow(['train_loss', 'test_loss'])
    with open(ACCURACIES_CSV, 'w') as accuracies_csv:
      accuracy_writer = csv.writer(accuracies_csv)
      accuracy_writer.writerow(['train_accuray', 'test_accuracy'])

  epoch = 0
  train_losses = []
  test_losses = []
  with open(LOSSES_CSV, 'r') as losses_csv:
    next(losses_csv)
    for row in losses_csv:
      epoch += 1
      train_loss, test_loss = eval(row)
      train_losses.append(train_loss)
      test_losses.append(test_loss)
  
  train_accuracies = []
  test_accuracies = []
  with open(ACCURACIES_CSV, 'r') as accuracies_csv:
    next(accuracies_csv)
    for row in accuracies_csv:
      train_accuracy, test_accuracy = eval(row)
      train_accuracies.append(train_accuracy)
      test_accuracies.append(test_accuracy)

  if os.path.isfile(MODEL_PATH):
    model = torch.load(MODEL_PATH)
  else:
    model = GeoNet()

  with open(LOSSES_CSV, 'a') as losses_csv, open(ACCURACIES_CSV, 'a') as accuracies_csv:
    loss_writer = csv.writer(losses_csv)
    accuracy_writer = csv.writer(accuracies_csv)

    while epoch < NUM_EPOCHS:
      epochs_per_lr = NUM_EPOCHS // len(LEARNING_RATES)
      learning_rate = LEARNING_RATES[min(epoch // epochs_per_lr, len(LEARNING_RATES))]
      optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=MOMENTUM)

      train_loss, train_accuracy = train(model, train_loader, optimizer)
      test_loss, test_accuracy = test(model, test_loader)

      train_losses.append(train_loss)
      test_losses.append(test_loss)
      loss_writer.writerow([train_loss, test_loss])

      train_accuracies.append(train_accuracy)
      test_accuracies.append(test_accuracy)
      accuracy_writer.writerow([train_accuracy, test_accuracy])

      torch.save(model, MODEL_PATH)

      plt.figure()
      plt.title('Loss vs. Epoch')
      plt.plot(range(epoch + 1), train_losses, label='Train')
      plt.plot(range(epoch + 1), test_losses, label='Test')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.legend()
      plt.savefig(LOSSES_PLOT)

      plt.figure()
      plt.title('Accuracy vs. Epoch')
      plt.plot(range(epoch + 1), train_accuracies, label='Train')
      plt.plot(range(epoch + 1), test_accuracies, label='Test')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.legend()
      plt.savefig(ACCURACIES_PLOT)

      print(f'epoch: {epoch + 1}/{NUM_EPOCHS}, train accuracy: {train_accuracy}, test accuracy: {test_accuracy}, time: {round(perf_counter() - start_time, 1)}s')
      epoch += 1

  print(f'done!')

if __name__ == '__main__':
  main()
