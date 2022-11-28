import requests
import random

RADIUS = 10000  # search radius in meters

with open('key.txt') as f:
  api_key = f.readlines()[0]

location_found = False
while not location_found:
  search_lat = random.uniform(-90, 90)
  search_lng = random.uniform(-180, 180)
  search_location = f'{search_lat},{search_lng}'
  metadata_url = 'https://maps.googleapis.com/maps/api/streetview/metadata'
  metadata_params = {
    'location': search_location,
    'radius': RADIUS,
    'key': api_key
  }
  metadata = requests.get(metadata_url, metadata_params).json()
  location_found = metadata['status'] == 'OK' and metadata['copyright'] == '© Google'

pano_id = metadata['pano_id']
lat = metadata['location']['lat']
lng = metadata['location']['lng']

fov = 120  # 0 - 120
heading = 0  # 0 - 360
pitch = 0  # -90 - 90

img_url = 'https://maps.googleapis.com/maps/api/streetview'
img_params = {
  'pano': pano_id,
  'size': '640x640',
  'fov': fov,
  'heading': heading,
  'pitch': pitch,
  'key': api_key
}
img_response = requests.get(img_url, img_params)
with open('test.jpg', 'wb') as file:
  file.write(img_response.content)

print(f'pano_id: {pano_id}')
print(f'location: {lat}, {lng}')
