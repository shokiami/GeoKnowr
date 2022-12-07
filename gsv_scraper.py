import requests
from playwright.sync_api import sync_playwright, TimeoutError
import os
import csv
import random
from time import perf_counter

NUM_IMAGES = 32000
IMAGE_WIDTH = 480
IMAGE_HEIGHT = 360

GSV_SCRAPER_OUT = 'gsv_scraper_out'
IMAGES_CSV = os.path.join(GSV_SCRAPER_OUT, 'images.csv')
IMAGES_DIR = os.path.join(GSV_SCRAPER_OUT, 'images')
API_KEY = 'key.txt'

start_time = perf_counter()

def main():
  if not os.path.isdir(GSV_SCRAPER_OUT):
    os.makedirs(GSV_SCRAPER_OUT)
    with open(IMAGES_CSV, 'w') as images_csv:
      writer = csv.writer(images_csv)
      writer.writerow(['pano_id', 'lat', 'lng'])
    os.makedirs(IMAGES_DIR)

  i = 0
  pano_ids = set()
  with open(IMAGES_CSV, 'r') as images_csv:
    next(images_csv)
    for row in images_csv:
      i += 1
      pano_id = row.split(',')[0]
      pano_ids.add(pano_id)

  with open(API_KEY, 'r') as api_key:
    key = api_key.read()

  with open(IMAGES_CSV, 'a') as images_csv, sync_playwright() as playwright, playwright.webkit.launch() as browser:
    writer = csv.writer(images_csv)
    context = browser.new_context(viewport={'width': IMAGE_WIDTH, 'height': IMAGE_HEIGHT})
    page = context.new_page()

    while i < NUM_IMAGES:
      try:
        location_found = False

        while not location_found:
          search_lat = random.uniform(-90, 90)
          search_lng = random.uniform(-180, 180)

          url = 'https://maps.googleapis.com/maps/api/streetview/metadata'
          params = {
            'location': f'{search_lat},{search_lng}',
            'radius': 10000,  # search radius in meters
            'key': key
          }

          metadata = requests.get(url, params).json()
          location_found = metadata['status'] == 'OK' and metadata['copyright'] == '© Google'

        pano_id = metadata['pano_id']
        lat = metadata['location']['lat']
        lng = metadata['location']['lng']
        heading = random.uniform(0, 360)

        if pano_id in pano_ids:
          continue

        url = f'https://www.google.com/maps/@{lat},{lng},3a,75y,{heading}h,90t/data=!3m6!1e1!3m4!1s{pano_id}!2e0!7i16384!8i8192'
        page.goto(url)

        page.wait_for_selector('canvas', timeout=5000)  # wait for canvas to load
        js_injection = """
          canvas = document.querySelector('canvas');
          context = canvas.getContext('webgl');
          if (context == null) {
            context = canvas.getContext('webgl2');
          }
          context.drawArrays = function() { }
        """
        page.evaluate_handle(js_injection)

        page.wait_for_selector('#minimap div div:nth-child(2)', timeout=5000)  # wait for image to load
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

        pano_ids.add(pano_id)
        writer.writerow([pano_id, lat, lng])
        page.screenshot(path=os.path.join(IMAGES_DIR, f'{pano_id}.png'))
        print(f'scraped: {i + 1}/{NUM_IMAGES}, time: {round(perf_counter() - start_time, 1)}s')
        i += 1

      except TimeoutError:
        pass

  print(f'done!')

if __name__ == '__main__':
  main()
