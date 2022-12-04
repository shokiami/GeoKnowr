import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import io
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

IMAGES_CSV = 'images2.csv'
IMAGES_DIR = 'images2'
BATCH_SIZE = 32
EPOCHS = 20
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
    self.conv1 = nn.Conv2d(3, 8, 3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
    self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
    self.conv6 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
    self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
    self.conv8 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)
    self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
    self.fc1 = nn.Linear(4096, 2)
    self.accuracy = None

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.maxpool(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = self.maxpool(x)
    x = self.conv3(x)
    x = F.relu(x)
    x = self.maxpool(x)
    x = self.conv4(x)
    x = F.relu(x)
    x = self.maxpool(x)
    x = self.conv5(x)
    x = F.relu(x)
    x = self.maxpool(x)
    x = self.conv6(x)
    x = F.relu(x)
    x = self.maxpool(x)
    x = self.conv7(x)
    x = F.relu(x)
    x = self.maxpool(x)
    x = self.conv8(x)
    x = F.relu(x)
    x = self.maxpool(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    return x

  def loss(self, pred, label):
    lat1 = torch.deg2rad(torch.index_select(pred, 1, torch.tensor([0])))
    lng1 = torch.deg2rad(torch.index_select(pred, 1, torch.tensor([1])))
    lat2 = torch.deg2rad(torch.index_select(label, 1, torch.tensor([0])))
    lng2 = torch.deg2rad(torch.index_select(label, 1, torch.tensor([1])))
    return torch.mean(torch.arccos(torch.sin(lat1) * torch.sin(lat2) + torch.cos(lat1) * torch.cos(lat2) * torch.cos(lng2 - lng1)))

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
    print(f"Training... Batch: {batch + 1}/{len(train_loader)}")
  return np.mean(losses)

def test(model, test_loader):
  model.eval()
  losses = []
  for batch, (images, labels) in enumerate(test_loader):
    with torch.no_grad():
      labels_pred = model(images)
    loss = model.loss(labels_pred, labels)
    losses.append(loss.item())
    print(f"Testing... Batch: {batch + 1}/{len(test_loader)}")
  return np.mean(losses)

def main():
  data = Dataset()
  train_data, test_data = random_split(data, [0.9, 0.1])
  train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
  
  model = GeoNet()
  optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
  train_losses = []
  test_losses = []
  for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, optimizer)
    test_loss = test(model, test_loader)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print(f"Epoch: {epoch + 1}, Train loss: {train_loss}, Test loss: {test_loss}")
  
  plt.figure(1)
  plt.plot(range(EPOCHS), train_losses, label="Train")
  plt.plot(range(EPOCHS), test_losses, label="Test")
  plt.title("Loss vs. Epoch")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig("losses.png")

main()
