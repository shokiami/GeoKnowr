from gsv_scraper import IMAGES_CSV, IMAGES_DIR, API_KEY
from train import NUM_CLASSES, MODEL_PATH, preprocess
import torch
from torchvision import io
from sklearn.mixture import GaussianMixture
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
    images_df = pd.read_csv(IMAGES_CSV)
    train_size = int(0.9 * len(images_df))
    train_df = images_df[:train_size]
    coords = train_df[['lat', 'lng']].to_numpy()
    self.gm = GaussianMixture(n_components=NUM_CLASSES, random_state=0).fit(coords)
    self.model = torch.load(MODEL_PATH)
    self.model.eval()

  def guess(self, image_path):
    image = io.read_image(image_path).float()
    if image.shape[0] == 4:  # remove alpha channel
      image = image[:3]
    image = preprocess(image)
    
    with torch.no_grad():
      pred = self.model(image.unsqueeze(0))

    cluster = torch.argmax(pred).item()
    pred_lat, pred_lng = self.gm.means_[cluster]

    return pred_lat, pred_lng

def distance(lat1, lng1, lat2, lng2):
  lat1 = np.deg2rad(lat1)
  lng1 = np.deg2rad(lng1)
  lat2 = np.deg2rad(lat2)
  lng2 = np.deg2rad(lng2)
  return 6371 * np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lng2 - lng1))

def visualize_clusters(train_df, key):
  coords = train_df[['lat', 'lng']].to_numpy()
  gm = GaussianMixture(n_components=NUM_CLASSES, random_state=0).fit(coords)
  clusters = gm.predict(coords)

  colors = ['0xff0000', '0xffa500', '0xffff00', '0x00ff00', '0x2b65ec', '0xe6e6fa', '0x00ffff', '0xffffff', '0xa00000', '0x008080', '0xc0c0c0',
            '0xffd700', '0xff4500', '0x964b00', '0xff9999', '0x023020', '0xff66ff', '0x00008b', '0x606060', '0xc4a484', '0xff007f']
  examples = []
  for cluster in range(NUM_CLASSES):
    count = 0
    i = 0
    while count < 200 // NUM_CLASSES:
      if clusters[i] == cluster:
        examples.append(i)
        count += 1
      i += 1

  url = 'https://maps.googleapis.com/maps/api/staticmap'
  params = {
    'markers': [f'color:{colors[clusters[i]]}|{coords[i][0]},{coords[i][1]}' for i in examples],
    'size': f'{MAP_WIDTH}x{MAP_HEIGHT}',
    'key': key
  }

  with open(os.path.join(TEST_OUT, f'clusters.png'), 'wb') as map_path:
    map_path.write(requests.get(url, params).content)

def main():
  if os.path.isdir(TEST_OUT):
    shutil.rmtree(TEST_OUT)
  os.makedirs(TEST_OUT)

  with open(API_KEY, 'r') as api_key:
    key = api_key.read()

  images_df = pd.read_csv(IMAGES_CSV)
  train_size = int(0.9 * len(images_df))
  train_df = images_df[:train_size]
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

  visualize_clusters(train_df, key)

if __name__ == '__main__':
  main()
