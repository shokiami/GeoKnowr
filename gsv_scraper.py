import requests
from playwright.sync_api import sync_playwright
import pandas as pd
import os
import shutil
import random
from time import perf_counter

NUM_IMAGES = 10
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
TARGET_DIR = 'images'

with open('key.txt') as f:
  api_key = f.readlines()[0]

if os.path.isdir(TARGET_DIR):
  shutil.rmtree(TARGET_DIR)
os.makedirs(TARGET_DIR)

start = perf_counter()

images = {'pano_id': [], 'lat': [], 'lng': []}

for i in range(NUM_IMAGES):
  location_found = False

  while not location_found:
    metadata_url = 'https://maps.googleapis.com/maps/api/streetview/metadata'

    search_lat = random.uniform(-90, 90)
    search_lng = random.uniform(-180, 180)
    search_location = f'{search_lat},{search_lng}'
    metadata_params = {
      'location': search_location,
      'radius': 10000,  # search radius in meters
      'key': api_key
    }

    metadata = requests.get(metadata_url, metadata_params).json()
    location_found = metadata['status'] == 'OK' and metadata['copyright'] == '© Google'
  
  pano_id = metadata['pano_id']
  lat = metadata['location']['lat']
  lng = metadata['location']['lng']

  images['pano_id'].append(pano_id)
  images['lat'].append(lat)
  images['lng'].append(lng)

  print(f'discovered {i+1}/{NUM_IMAGES}: {round(perf_counter() - start, 1)}s')

images_df = pd.DataFrame(images)
images_df.to_csv('images.csv', index=False)
print(f'saved images.csv!')

with sync_playwright() as playwright:
  webkit = playwright.webkit
  browser = webkit.launch()
  context = browser.new_context(viewport={'width': IMAGE_WIDTH, 'height': IMAGE_HEIGHT})
  page = context.new_page()

  for i in range(NUM_IMAGES):
    image = images_df.iloc[i]
    pano_id = image['pano_id']
    lat = image['lat']
    lng = image['lng']

    gsv_url = f'https://www.google.com/maps/@{lat},{lng},3a,75y,0h,90t/data=!3m6!1e1!3m4!1s{pano_id}!2e0!7i16384!8i8192'
    page.goto(gsv_url)

    page.wait_for_selector('canvas')  # wait for canvas to load
    js_injection = """
      canvas = document.querySelector('canvas');
      context = canvas.getContext('webgl');
      context.drawArrays = function() { }
    """
    page.evaluate_handle(js_injection)

    page.wait_for_selector('#minimap div div:nth-child(2)')  # wait for image to load
    elements_to_hide = """
      .app-viewcard-strip,
      .scene-footer,
      #titlecard,
      #watermark,
      #image-header {
        display: none;
      }
    """
    page.add_style_tag(content=elements_to_hide)

    page.screenshot(path=f'{TARGET_DIR}/{pano_id}.png')

    print(f'downloaded {i+1}/{NUM_IMAGES}: {round(perf_counter() - start, 1)}s')
  
  browser.close()

print(f'finished!')
