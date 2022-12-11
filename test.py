from gsv_scraper import IMAGES_CSV, IMAGES_DIR, API_KEY
from train import NUM_CLASSES, MODEL_PATH, preprocess
import torch
import torchvision.io
from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull
import numpy as np
import requests
import os
import shutil
import pandas as pd
import random

NUM_EXAMPLES = 10
MAP_WIDTH = 960
MAP_HEIGHT = 720

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
    image = torchvision.io.read_image(image_path).float()
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

  colors = ['0xe6194b', '0x3cb44b', '0xffe119', '0x4363d8', '0xfffac8', '0x911eb4', '0x42d4f4',
            '0xffffff', '0xaaffc3', '0xfabed4', '0x469990', '0xdcbeff', '0x4b0082', '0x800000',
            '0xffd8b1', '0x808000', '0xf58231', '0x000075', '0xf032e6', '0xa0a0a0', '0x000000']

  points = []
  for cluster in range(NUM_CLASSES):
    points.append([])
  for i in range(len(coords)):
    lat, lng = coords[i]
    center_lat, center_lng = gm.means_[clusters[i]]
    if distance(lat, lng, center_lat, center_lng) < 4000:
      points[clusters[i]].append((lat, lng))
  paths = []
  for cluster in range(NUM_CLASSES):
    indices = ConvexHull(points[cluster]).vertices
    paths.append('')
    for i in indices:
      lat, lng = points[cluster][i]
      paths[cluster] += f'{lat},{lng}|'
    lat, lng = points[cluster][indices[0]]
    paths[cluster] += f'{lat},{lng}'

  url = 'https://maps.googleapis.com/maps/api/staticmap'
  params = {
    'path': [f'color:0x00000000|fillcolor:{colors[i]}88|{paths[i]}' for i in range(NUM_CLASSES)],
    'zoom': 1,
    'scale': 2,
    'size': f'{int(MAP_WIDTH / 2)}x{int(MAP_HEIGHT / 2)}',
    'key': key
  }

  with open(os.path.join(TEST_OUT, 'clusters.png'), 'wb') as map_path:
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
      'scale': 2,
      'size': f'{int(MAP_WIDTH / 2)}x{int(MAP_HEIGHT / 2)}',
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
