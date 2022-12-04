import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import io
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter

IMAGES_CSV = 'images.csv'
IMAGES_DIR = 'images'
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9

class Dataset(torch.utils.data.Dataset):
  def __init__(self):
    super(Dataset, self).__init__()
    self.images_df = pd.read_csv(IMAGES_CSV)

  def __len__(self):
    return len(self.images_df)

  def __getitem__(self, i):
    image = self.images_df.iloc[i]
    pano_id = image['pano_id']
    lat = image['lat']
    lng = image['lng']

    image_path = f"{IMAGES_DIR}/{pano_id}.png"
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
    return torch.mean(torch.arccos(torch.sin(lat1) * torch.sin(lat2) + torch.cos(lat1) * torch.cos(lat2) * torch.cos(lng2 - lng1)))
    # dlat = torch.abs(lat1 - lat2)
    # dlng = torch.min(torch.remainder(lng1 - lng2, 360), torch.remainder(lng2 - lng1, 360))
    # return torch.mean(dlat + dlng)

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
    print(f"batch: {batch + 1}/{len(train_loader)}, train loss: {loss.item()}")
  return np.mean(losses)

def test(model, test_loader):
  model.eval()
  losses = []
  with torch.no_grad():
    for batch, (images, labels) in enumerate(test_loader):
      labels_pred = model(images)
      loss = model.loss(labels_pred, labels)
      losses.append(loss.item())
  return np.mean(losses)

def main():
  start = perf_counter()  # start timer

  data = Dataset()
  train_data, test_data = random_split(data, [0.9, 0.1])
  train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
  
  model = GeoNet()
  optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
  train_losses = []
  test_losses = []
  for epoch in range(NUM_EPOCHS):
    train_loss = train(model, train_loader, optimizer)
    test_loss = test(model, test_loader)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f"epoch: {epoch + 1}, train loss: {train_loss}, test loss: {test_loss}, time: {round(perf_counter() - start, 1)}s")
  
  plt.figure(1)
  plt.plot(range(NUM_EPOCHS), train_losses, label="Train")
  plt.plot(range(NUM_EPOCHS), test_losses, label="Test")
  plt.title("Loss vs. Epoch")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig("losses.png")

  print(f"done!")

main()
