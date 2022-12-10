from gsv_scraper import IMAGES_CSV, IMAGES_DIR, API_KEY
from train import MODEL_PATH, preprocess
import torch
from torchvision import io
import numpy as np
import requests
import os
import shutil
import pandas as pd
import random

NUM_EXAMPLES = 10
MAP_WIDTH = 480
MAP_HEIGHT = 360

TEST_OUT = 'test_out'

class GeoKnowr():
  def __init__(self):
    self.model = torch.load(MODEL_PATH)
    self.model.eval()

  def guess(self, image_path):
    image = io.read_image(image_path).float()
    if image.shape[0] == 4:  # remove alpha channel
      image = image[:3]
    image = preprocess(image)

    with torch.no_grad():
      pred = self.model(image.unsqueeze(0))[0].numpy()
    pred_lat = 90 * pred[0]
    pred_lng = 180 * pred[1]

    return pred_lat, pred_lng

def distance(lat1, lng1, lat2, lng2):
  lat1 = np.deg2rad(lat1)
  lng1 = np.deg2rad(lng1)
  lat2 = np.deg2rad(lat2)
  lng2 = np.deg2rad(lng2)
  return 6371 * np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lng2 - lng1))

def main():
  if os.path.isdir(TEST_OUT):
    shutil.rmtree(TEST_OUT)
  os.makedirs(TEST_OUT)

  with open(API_KEY, 'r') as api_key:
    key = api_key.read()

  images_df = pd.read_csv(IMAGES_CSV)
  train_size = int(0.9 * len(images_df))
  test_df = images_df[train_size:]
  vis_df = test_df.iloc[random.sample(range(len(test_df)), NUM_EXAMPLES)]

  geo_knowr = GeoKnowr()

  for i in range(len(vis_df)):
    pano_id, lat, lng = vis_df.iloc[i]

    image_path = os.path.join(TEST_OUT, f'image_{i + 1}.png')
    shutil.copyfile(os.path.join(IMAGES_DIR, f'{pano_id}.png'), image_path)

    pred_lat, pred_lng = geo_knowr.guess(image_path)

    dist = distance(pred_lat, pred_lng, lat, lng)

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
    print(f'distance (km): {dist}')
    print()

if __name__ == '__main__':
  main()
