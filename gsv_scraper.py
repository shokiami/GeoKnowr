import requests
from playwright.sync_api import sync_playwright, TimeoutError
import csv
import os
import random
from time import perf_counter

NUM_IMAGES = 10000
IMAGE_WIDTH = 480
IMAGE_HEIGHT = 360
GSV_SCRAPER_OUT = 'gsv_scraper_out'
IMAGES_CSV = os.path.join(GSV_SCRAPER_OUT, 'images.csv')
IMAGES_DIR = os.path.join(GSV_SCRAPER_OUT, 'images')
API_KEY = 'key.txt'

def main():
  start = perf_counter()  # start timer

  if not os.path.isdir(GSV_SCRAPER_OUT):
    os.makedirs(GSV_SCRAPER_OUT)
    with open(IMAGES_CSV, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(['pano_id', 'lat', 'lng'])
    os.makedirs(IMAGES_DIR)

  with open(IMAGES_CSV, 'r') as f:
    next(f)
    prev_scraped = sum(1 for row in f)

  with open(IMAGES_CSV, 'a') as f, sync_playwright() as playwright, playwright.webkit.launch() as browser:
    writer = csv.writer(f)
    context = browser.new_context(viewport={'width': IMAGE_WIDTH, 'height': IMAGE_HEIGHT})
    page = context.new_page()

    for i in range(prev_scraped, NUM_IMAGES):
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
      heading = random.uniform(0, 360)

      gsv_url = f'https://www.google.com/maps/@{lat},{lng},3a,75y,{heading}h,90t/data=!3m6!1e1!3m4!1s{pano_id}!2e0!7i16384!8i8192'
      page.goto(gsv_url)

      try:
        page.wait_for_selector('canvas', timeout=5000)  # wait for canvas to load
      except TimeoutError:
        continue
      js_injection = """
        canvas = document.querySelector('canvas');
        context = canvas.getContext('webgl');
        if (context == null) {
          context = canvas.getContext('webgl2');
        }
        context.drawArrays = function() { }
      """
      page.evaluate_handle(js_injection)

      try:
        page.wait_for_selector('#minimap div div:nth-child(2)', timeout=5000)  # wait for image to load
      except TimeoutError:
        continue
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

      writer.writerow([pano_id, lat, lng])
      page.screenshot(path=os.path.join(IMAGES_DIR, f'{pano_id}.png'))
      print(f'scraped: {i + 1}/{NUM_IMAGES}, time: {round(perf_counter() - start, 1)}s')

  print(f'done!')

main()
