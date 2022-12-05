import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import io
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter

GSV_SCRAPER_OUT = 'gsv_scraper_out'
IMAGES_CSV = os.path.join(GSV_SCRAPER_OUT, 'images.csv')
IMAGES_DIR = os.path.join(GSV_SCRAPER_OUT, 'images')
TRAIN_OUT = 'train_out'
MODEL_PATH = os.path.join(TRAIN_OUT, 'model.pt')
LOSSES_CSV = os.path.join(TRAIN_OUT, 'losses.csv')
PLOT_PATH = os.path.join(TRAIN_OUT, 'losses.png')
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9

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

    image_path = f'{IMAGES_DIR}/{pano_id}.png'
    image = io.read_image(image_path).float()

    if image.shape[0] == 4:  # remove alpha channel
      image = image[:3]

    label = torch.FloatTensor([lat, lng])

    return image, label

class GeoNet(nn.Module):
  def __init__(self):
    super(GeoNet, self).__init__()

    self.num_conv_blocks = 7

    self.conv_layers = []
    for i in range(self.num_conv_blocks):
      if i == 0:
        self.conv_layers.append(nn.Conv2d(3, 8, 3, stride=1, padding=1))
      else:
        self.conv_layers.append(nn.Conv2d(2**(i + 2), 2**(i + 3), 3, stride=1, padding=1))
    
    self.bn_layers = []
    for i in range(self.num_conv_blocks):
      self.bn_layers.append(nn.BatchNorm2d(2**(i + 3)))

    self.mp = nn.MaxPool2d(3, stride=2, padding=1)

    self.fc = nn.Linear(6144, 2)

  def forward(self, x):
    for i in range(self.num_conv_blocks):
      x = self.conv_layers[i](x)
      x = self.bn_layers[i](x)
      x = F.relu(x)
      x = self.mp(x)

    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x

  def loss(self, pred, label):
    lat1 = torch.deg2rad(torch.index_select(pred, 1, torch.tensor([0])))
    lng1 = torch.deg2rad(torch.index_select(pred, 1, torch.tensor([1])))
    lat2 = torch.deg2rad(torch.index_select(label, 1, torch.tensor([0])))
    lng2 = torch.deg2rad(torch.index_select(label, 1, torch.tensor([1])))

    dlat = lat1 - lat2
    dlng = torch.min(torch.remainder(lng1 - lng2, 360), torch.remainder(lng2 - lng1, 360))
    return torch.mean(dlat * dlat + dlng * dlng)

    # return torch.mean(torch.arccos(torch.sin(lat1) * torch.sin(lat2) + torch.cos(lat1) * torch.cos(lat2) * torch.cos(lng2 - lng1)))  # mean distance on unit sphere

def train(model, train_loader, optimizer):
  model.train()
  losses = []
  for batch, (images, labels) in enumerate(train_loader):
    optimizer.zero_grad()
    labels_pred = model(images)
    loss = model.loss(labels_pred, labels)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f'batch: {batch + 1}/{len(train_loader)}, train loss: {loss.item()}')
  return np.mean(losses)

def test(model, test_loader):
  print('testing...')
  model.eval()
  losses = []
  with torch.no_grad():
    for batch, (images, labels) in enumerate(test_loader):
      labels_pred = model(images)
      loss = model.loss(labels_pred, labels)
      losses.append(loss.item())
  return np.mean(losses)

if __name__ ==  "__main__":
  start = perf_counter()  # start timer

  images_df = pd.read_csv(IMAGES_CSV)
  train_size = int(0.9 * len(images_df))
  train_data = GeoData(images_df[:train_size])
  test_data = GeoData(images_df[train_size:])
  train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

  if not os.path.isdir(TRAIN_OUT):
    os.makedirs(TRAIN_OUT)
    losses_df = pd.DataFrame(columns=['train_losse', 'test_losse'])
    losses_df.to_csv(LOSSES_CSV, index=False)

  losses_df = pd.read_csv(LOSSES_CSV)
  
  if os.path.isfile(MODEL_PATH):
    model = torch.load(MODEL_PATH)
  else:
    model = GeoNet()

  optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

  for epoch in range(len(losses_df), NUM_EPOCHS):
    train_loss = train(model, train_loader, optimizer)
    test_loss = test(model, test_loader)

    torch.save(model, MODEL_PATH)
    losses_df.loc[len(losses_df)] = [train_loss, test_loss]
    losses_df.to_csv(LOSSES_CSV, index=False)

    plt.figure()
    plt.title('Loss vs. Epoch')
    plt.plot(range(epoch + 1), losses_df['train_loss'].to_list(), label='Train')
    plt.plot(range(epoch + 1), losses_df['test_loss'].to_list(), label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(PLOT_PATH)

    print(f'epoch: {epoch + 1}/{NUM_EPOCHS}, train loss: {train_loss}, test loss: {test_loss}, time: {round(perf_counter() - start, 1)}s')

  print(f'done!')
