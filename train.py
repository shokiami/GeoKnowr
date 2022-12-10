from gsv_scraper import IMAGES_CSV, IMAGES_DIR
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import io
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter

NUM_EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATES = [0.001, 0.0005, 0.0001]
WEIGHT_DECAY = 0.0001

TRAIN_OUT = 'train_out'
MODEL_PATH = os.path.join(TRAIN_OUT, 'model.pt')
LOSSES_CSV = os.path.join(TRAIN_OUT, 'losses.csv')
LOSSES_PLOT = os.path.join(TRAIN_OUT, 'losses.png')

start_time = perf_counter()

def preprocess(image):
  weights = ResNet18_Weights.DEFAULT
  transform = weights.transforms()
  return transform(image)

class GeoData(Dataset):
  def __init__(self, images_df):
    super(Dataset, self).__init__()
    self.images_df = images_df

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

    image = preprocess(image)
    label = torch.FloatTensor([lat / 90, lng / 180])

    return image, label

def loss_function(pred, label):
  lat1 = torch.deg2rad(90 * pred[:,0])
  lng1 = torch.deg2rad(180 * pred[:,1])
  lat2 = torch.deg2rad(90 * label[:,0])
  lng2 = torch.deg2rad(180 * label[:,1])
  # mean arc distance over the unit sphere
  dist = torch.arccos(torch.sin(lat1) * torch.sin(lat2) + torch.cos(lat1) * torch.cos(lat2) * torch.cos(lng2 - lng1))
  return torch.mean(dist)

def train(model, train_loader, optimizer):
  print('training...')
  model.train()
  losses = []
  for batch, (images, labels) in enumerate(train_loader):
    optimizer.zero_grad()
    pred = model(images)
    loss = loss_function(pred, labels)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f'batch: {batch + 1}/{len(train_loader)}, train loss: {loss.item()}, time: {round(perf_counter() - start_time, 1)}s')
  train_loss = np.mean(losses)
  return train_loss

def test(model, test_loader):
  print('testing...')
  model.eval()
  losses = []
  with torch.no_grad():
    for batch, (images, labels) in enumerate(test_loader):
      labels = labels.squeeze()
      pred = model(images)
      loss = loss_function(pred, labels)
      losses.append(loss.item())
      print(f'batch: {batch + 1}/{len(test_loader)}, test loss: {loss.item()}, time: {round(perf_counter() - start_time, 1)}s')
  test_loss = np.mean(losses)
  return test_loss

def main():
  images_df = pd.read_csv(IMAGES_CSV)
  train_size = int(0.9 * len(images_df))
  train_df = images_df[:train_size]
  test_df = images_df[train_size:]

  train_data = GeoData(train_df)
  test_data = GeoData(test_df)
  train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

  if not os.path.isdir(TRAIN_OUT):
    os.makedirs(TRAIN_OUT)
    with open(LOSSES_CSV, 'w') as losses_csv:
      loss_writer = csv.writer(losses_csv)
      loss_writer.writerow(['train_loss', 'test_loss'])

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

  if os.path.isfile(MODEL_PATH):
    model = torch.load(MODEL_PATH)
  else:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # only train final layer
    for param in model.parameters():
      param.requires_grad = False
    # re-initialize final fully-connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.add_module('tanh', nn.Tanh())

  with open(LOSSES_CSV, 'a') as losses_csv:
    loss_writer = csv.writer(losses_csv)

    while epoch < NUM_EPOCHS:
      epochs_per_lr = NUM_EPOCHS // len(LEARNING_RATES)
      learning_rate = LEARNING_RATES[min(epoch // epochs_per_lr, len(LEARNING_RATES) - 1)]

      # optimizer only for training final fully-connected layer
      optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)

      train_loss = train(model, train_loader, optimizer)
      test_loss = test(model, test_loader)

      train_losses.append(train_loss)
      test_losses.append(test_loss)
      loss_writer.writerow([train_loss, test_loss])

      torch.save(model, MODEL_PATH)

      plt.figure()
      plt.title('Loss vs. Epoch')
      plt.plot(range(epoch + 1), train_losses, label='Train')
      plt.plot(range(epoch + 1), test_losses, label='Test')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.legend()
      plt.savefig(LOSSES_PLOT)

      print(f'epoch: {epoch + 1}/{NUM_EPOCHS}, train loss: {train_loss}, test loss: {test_loss}, time: {round(perf_counter() - start_time, 1)}s')
      epoch += 1

  print(f'done!')

if __name__ == '__main__':
  main()
