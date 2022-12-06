from train import GeoNet
import torch
from torchvision import io
import requests
import os
import shutil
import pandas as pd
import random

NUM_EXAMPLES = 5
EARTH_RADIUS = 6371  # in kilometers
MAP_WIDTH = 480
MAP_HEIGHT = 360

GSV_SCRAPER_OUT = 'gsv_scraper_out'
IMAGES_CSV = os.path.join(GSV_SCRAPER_OUT, 'images.csv')
IMAGES_DIR = os.path.join(GSV_SCRAPER_OUT, 'images')
TRAIN_OUT = 'train_out'
MODEL_PATH = os.path.join(TRAIN_OUT, 'model.pt')
LOSSES_CSV = os.path.join(TRAIN_OUT, 'losses.csv')
PLOT_PATH = os.path.join(TRAIN_OUT, 'losses.png')
TEST_OUT = 'test_out'
API_KEY = 'key.txt'

def main():
  shutil.rmtree(TEST_OUT)
  os.makedirs(TEST_OUT)

  with open(API_KEY, 'r') as api_key:
    key = api_key.read()

  images_df = pd.read_csv(IMAGES_CSV)
  train_size = int(0.9 * len(images_df))
  images_df = images_df[train_size:]  # only want images from test set
  images_df = images_df.iloc[random.sample(range(len(images_df)), NUM_EXAMPLES)]

  model = torch.load(MODEL_PATH)
  model.eval()

  for i in range(len(images_df)):
    pano_id, lat, lng = images_df.iloc[i]

    image_path = os.path.join(TEST_OUT, f'image_{i + 1}.png')
    shutil.copyfile(os.path.join(IMAGES_DIR, f'{pano_id}.png'), image_path)

    image = io.read_image(image_path).float()
    if image.shape[0] == 4:  # remove alpha channel
      image = image[:3]
    image = image.unsqueeze(0)
    label = torch.FloatTensor([lat, lng]).unsqueeze(0)
    
    with torch.no_grad():
      pred = model(image)

    pred_lat = torch.index_select(pred, 1, torch.tensor([0])).item()
    pred_lng = torch.index_select(pred, 1, torch.tensor([1])).item()

    dist = EARTH_RADIUS * model.loss(pred, label).item()

    url = 'https://maps.googleapis.com/maps/api/staticmap'
    params = {
      'markers': [f'{lat},{lng}', f'color:black|label:G|{pred_lat},{pred_lng}'],
      'size': f'{MAP_WIDTH}x{MAP_HEIGHT}',
      'key': key
    }

    with open(os.path.join(TEST_OUT, f'map_{i + 1}.png'), 'wb') as map_path:
      map_path.write(requests.get(url, params).content)

    print(f'example {i + 1}/{NUM_EXAMPLES}:')
    print(f'actual: {lat}, {lng}')
    print(f'guess: {pred_lat}, {pred_lng}')
    print(f'distance: {dist}km')
    print()

if __name__ == '__main__':
  main()
