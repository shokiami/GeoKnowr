import requests
from playwright.sync_api import sync_playwright
import pandas as pd
import os
import shutil
import random
from time import perf_counter

NUM_IMAGES = 640
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
IMAGES_CSV = 'images.csv'
IMAGES_DIR = 'images'
API_KEY = 'key.txt'

# start timer
start = perf_counter()

# finds random GSV locations and outputs them to IMAGES_CSV
def find_locations():
  images = {'pano_id': [], 'lat': [], 'lng': []}

  for i in range(NUM_IMAGES):
    location_found = False

    while not location_found:
      search_lat = random.uniform(-90, 90)
      search_lng = random.uniform(-180, 180)

      metadata_url = 'https://maps.googleapis.com/maps/api/streetview/metadata'
      metadata_params = {
        'location': f'{search_lat},{search_lng}',
        'radius': 10000,  # search radius in meters
        'key': open(API_KEY).read()
      }

      metadata = requests.get(metadata_url, metadata_params).json()
      location_found = metadata['status'] == 'OK' and metadata['copyright'] == '© Google'

    pano_id = metadata['pano_id']
    lat = metadata['location']['lat']
    lng = metadata['location']['lng']

    images['pano_id'].append(pano_id)
    images['lat'].append(lat)
    images['lng'].append(lng)

    print(f'found {i + 1}/{NUM_IMAGES}: {round(perf_counter() - start, 1)}s')

  images_df = pd.DataFrame(images)
  images_df.to_csv(IMAGES_CSV, index=False)

  print('finished finding locations!')

# scrapes GSV images corresponding to locations in IMAGES_CSV
def scrape_images():
  images_df = pd.read_csv(IMAGES_CSV)

  if os.path.isdir(IMAGES_DIR):
    shutil.rmtree(IMAGES_DIR)
  os.makedirs(IMAGES_DIR)

  with sync_playwright() as playwright:
    browser = playwright.webkit.launch()
    context = browser.new_context(viewport={'width': IMAGE_WIDTH, 'height': IMAGE_HEIGHT})
    page = context.new_page()

    for i in range(len(images_df)):
      image = images_df.iloc[i]
      pano_id = image['pano_id']
      lat = image['lat']
      lng = image['lng']

      heading = random.uniform(0, 360)

      gsv_url = f'https://www.google.com/maps/@{lat},{lng},3a,75y,{heading}h,90t/data=!3m6!1e1!3m4!1s{pano_id}!2e0!7i16384!8i8192'
      page.goto(gsv_url)

      page.wait_for_selector('canvas')  # wait for canvas to load
      js_injection = """
        canvas = document.querySelector('canvas');
        context = canvas.getContext('webgl');
        if (context == null) {
          context = canvas.getContext('webgl2');
        }
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

      page.screenshot(path=f'{IMAGES_DIR}/{pano_id}.png')

      print(f'scraped {i + 1}/{len(images_df)}: {round(perf_counter() - start, 1)}s')

    browser.close()

  print('finished scraping images!')

find_locations()
scrape_images()
