import requests
from playwright.sync_api import sync_playwright, TimeoutError
import pandas as pd
import os
import random
import multiprocessing as mp
from time import perf_counter

NUM_IMAGES = 10000
IMAGE_WIDTH = 480
IMAGE_HEIGHT = 360
GSV_SCRAPER_OUT = 'gsv_scraper_out'
IMAGES_CSV = os.path.join(GSV_SCRAPER_OUT, 'images.csv')
IMAGES_DIR = os.path.join(GSV_SCRAPER_OUT, 'images')
API_KEY = 'key.txt'
NUM_THREADS = min(mp.cpu_count(), 8)

def scrape_image(num_images, queue):
  with sync_playwright() as playwright, playwright.webkit.launch() as browser:
    context = browser.new_context(viewport={'width': IMAGE_WIDTH, 'height': IMAGE_HEIGHT})
    page = context.new_page()

    for i in range(num_images):
      try:
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

        page.screenshot(path=os.path.join(IMAGES_DIR, f'{pano_id}.png'))

        queue.put([pano_id, lat, lng])

      except TimeoutError:
        pass

  queue.put('done')

if __name__ ==  "__main__":
  start = perf_counter()  # start timer

  if not os.path.isdir(GSV_SCRAPER_OUT):
    os.makedirs(GSV_SCRAPER_OUT)
    images_df = pd.DataFrame(columns=['pano_id', 'lat', 'lng'])
    images_df.to_csv(IMAGES_CSV, index=False)
    os.makedirs(IMAGES_DIR)

  images_df = pd.read_csv(IMAGES_CSV)

  left_to_scrape = NUM_IMAGES - len(images_df)
  num_images = [left_to_scrape // NUM_THREADS for i in range(NUM_THREADS)]
  extra = left_to_scrape % NUM_THREADS
  for i in range(extra):
    num_images[i] += 1

  queue = mp.Queue()
  processes = []
  for i in range(NUM_THREADS):
    process = mp.Process(target=scrape_image, args=(num_images[i], queue))
    processes.append(process)

  for process in processes:
    process.start()

  done_count = 0
  while done_count < NUM_THREADS:
    result = queue.get()
    if result == 'done':
      done_count += 1
    else:
      images_df.loc[len(images_df)] = result
      images_df.to_csv(IMAGES_CSV, index=False)
      print(f'scraped: {len(images_df)}/{NUM_IMAGES}, time: {round(perf_counter() - start, 1)}s')

  for process in processes:
    process.join()

  print(f'done!')
