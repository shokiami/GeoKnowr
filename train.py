import torch
import pandas as pd
from torchvision import io
from torch.utils.data import DataLoader

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

data = Dataset()
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

