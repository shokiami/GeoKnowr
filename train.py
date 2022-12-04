import torch
import pandas as pd
from torchvision import io
from torch.utils.data import DataLoader, random_split

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
train_data, test_data = random_split(data, [int(0.9 * len(data)), int(0.1 * len(data))])
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
