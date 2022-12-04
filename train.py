import torch
import torch.nn as nn
import pandas as pd
from torchvision import io
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

IMAGES_CSV = 'images.csv'
IMAGES_DIR = 'images'
BATCH_SIZE = 64

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
    image = io.read_image(image_path)
    return image, (lat, lng)

class GeoNet(nn.Module):
    def __init__(self):
        super(GeoNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.fc1 = nn.Linear(2048, 10)
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
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

def main():
  data = Dataset()
  train_data, test_data = random_split(data, [int(0.9 * len(data)), int(0.1 * len(data))])
  train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
  test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

main()
