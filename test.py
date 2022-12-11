from gsv_scraper import IMAGES_CSV, IMAGES_DIR, API_KEY
from demo import GeoKnowr
from scipy.spatial import ConvexHull
import numpy as np
import requests
import os
import shutil
import pandas as pd
import random
import matplotlib.pyplot as plt

NUM_EXAMPLES = 10
MAP_WIDTH = 960
MAP_HEIGHT = 720

TEST_OUT = 'test_out'

def distance(lat1, lng1, lat2, lng2):
  lat1 = np.deg2rad(lat1)
  lng1 = np.deg2rad(lng1)
  lat2 = np.deg2rad(lat2)
  lng2 = np.deg2rad(lng2)
  return 6371 * np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lng2 - lng1))

def visualize_clusters(geo_knowr, train_df, key):
  print('visualizing clusters...')

  coords = train_df[['lat', 'lng']].to_numpy()
  clusters = geo_knowr.gm.predict(coords)

  colors = ['0xe6194b', '0x3cb44b', '0xffe119', '0x4363d8', '0xfffac8', '0x911eb4', '0x42d4f4',
            '0xffffff', '0xaaffc3', '0xfabed4', '0x469990', '0xdcbeff', '0x4b0082', '0x800000',
            '0xffd8b1', '0x808000', '0xf58231', '0x000075', '0xf032e6', '0xa0a0a0', '0x000000']

  points = []
  for cluster in range(geo_knowr.num_classes):
    points.append([])
  for i in range(len(coords)):
    lat, lng = coords[i]
    center_lat, center_lng = geo_knowr.gm.means_[clusters[i]]
    if distance(lat, lng, center_lat, center_lng) < 4000:
      points[clusters[i]].append((lat, lng))

  paths = []
  for cluster in range(geo_knowr.num_classes):
    indices = ConvexHull(points[cluster]).vertices
    paths.append('')
    for i in indices:
      lat, lng = points[cluster][i]
      paths[cluster] += f'{lat},{lng}|'
    lat, lng = points[cluster][indices[0]]
    paths[cluster] += f'{lat},{lng}'

  url = 'https://maps.googleapis.com/maps/api/staticmap'
  params = {
    'path': [f'color:0x00000000|fillcolor:{colors[i]}88|{paths[i]}' for i in range(geo_knowr.num_classes)],
    'zoom': 1,
    'scale': 2,
    'size': f'{int(MAP_WIDTH / 2)}x{int(MAP_HEIGHT / 2)}',
    'key': key
  }

  with open(os.path.join(TEST_OUT, 'clusters.png'), 'wb') as clusters_path:
    clusters_path.write(requests.get(url, params).content)

def visualize_examples(geo_knowr, test_df, key):
  print('visualizing examples...')

  vis_df = test_df.iloc[random.sample(range(len(test_df)), NUM_EXAMPLES)]

  for i in range(len(vis_df)):
    pano_id, lat, lng = vis_df.iloc[i]

    image_path = os.path.join(IMAGES_DIR, f'{pano_id}.png')
    shutil.copyfile(image_path, os.path.join(TEST_OUT, f'image_{i}.png'))

    pred_lat, pred_lng = geo_knowr.guess(image_path)

    dist = distance(pred_lat, pred_lng, lat, lng)

    url = 'https://maps.googleapis.com/maps/api/staticmap'
    params = {
      'markers': [f'{lat},{lng}', f'color:black|label:G|{pred_lat},{pred_lng}'],
      'scale': 2,
      'size': f'{int(MAP_WIDTH / 2)}x{int(MAP_HEIGHT / 2)}',
      'key': key
    }

    with open(os.path.join(TEST_OUT, f'map_{i}.png'), 'wb') as map_path:
      map_path.write(requests.get(url, params).content)

    print(f'example {i + 1}/{NUM_EXAMPLES}:')
    print(f'ground truth (lat, lng): {lat}, {lng}')
    print(f'guess (lat, lng): {pred_lat}, {pred_lng}')
    print(f'distance (km): {dist}')
    print()

def plot_distances(geo_knowr, test_df):
  print(f'testing on {len(test_df)} examples...')
  distances = []
  for i in range(len(test_df)):
    pano_id, lat, lng = test_df.iloc[i]
    image_path = os.path.join(IMAGES_DIR, f'{pano_id}.png')
    pred_lat, pred_lng = geo_knowr.guess(image_path)
    distances.append(distance(pred_lat, pred_lng, lat, lng))
  plt.figure()
  plt.title('Guess Distance Distribution (3200 examples)')
  plt.hist(distances, bins=100, color='#4363d8')
  plt.xlabel('Distance (km) from Ground Truth')
  plt.ylabel('Number of Guesses')
  plt.savefig(os.path.join(TEST_OUT, 'distances.png'))
  print('done!')
  print()

def main():
  if os.path.isdir(TEST_OUT):
    shutil.rmtree(TEST_OUT)
  os.makedirs(TEST_OUT)

  geo_knowr = GeoKnowr()

  images_df = pd.read_csv(IMAGES_CSV)
  train_size = int(0.9 * len(images_df))
  train_df = images_df[:train_size]
  test_df = images_df[train_size:]

  with open(API_KEY, 'r') as api_key:
    key = api_key.read()

  visualize_clusters(geo_knowr, train_df, key)
  visualize_examples(geo_knowr, test_df, key)
  plot_distances(geo_knowr, test_df)

if __name__ == '__main__':
  main()
