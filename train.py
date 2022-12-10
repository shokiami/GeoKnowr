from gsv_scraper import IMAGES_CSV, IMAGES_DIR
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import io
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.mixture import GaussianMixture
import numpy as np
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter

NUM_CLASSES = 20
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATES = [0.001, 0.0005, 0.0001]
WEIGHT_DECAY = 0.0001

TRAIN_OUT = 'train_out'
MODEL_PATH = os.path.join(TRAIN_OUT, 'model.pt')
LOSSES_CSV = os.path.join(TRAIN_OUT, 'losses.csv')
ACCURACIES_CSV = os.path.join(TRAIN_OUT, 'accuracies.csv')
LOSSES_PLOT = os.path.join(TRAIN_OUT, 'losses.png')
ACCURACIES_PLOT = os.path.join(TRAIN_OUT, 'accuracies.png')

start_time = perf_counter()

def preprocess(image):
  weights = ResNet18_Weights.DEFAULT
  transform = weights.transforms()
  return transform(image)

class GeoData(Dataset):
  def __init__(self, images_df, gm):
    super(Dataset, self).__init__()
    self.images_df = images_df
    self.gm = gm

  def __len__(self):
    return len(self.images_df)

  def __getitem__(self, i):
    image = self.images_df.iloc[i]
    pano_id = image['pano_id']
    lat = image['lat']
    lng = image['lng']

    image_path = os.path.join(IMAGES_DIR, f'{pano_id}.png')
    image = io.read_image(image_path).float()
    if image.shape[0] == 4:  # remove alpha channel
      image = image[:3]

    image = preprocess(image)
    label = torch.LongTensor(self.gm.predict([(lat, lng)]))

    return image, label

def train(model, train_loader, optimizer):
  print('training...')
  model.train()
  losses = []
  accuracies = []
  for batch, (images, labels) in enumerate(train_loader):
    labels = labels.squeeze()
    optimizer.zero_grad()
    pred = model(images)
    loss = F.cross_entropy(pred, labels)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    accuracy = torch.sum(torch.argmax(pred, 1) == labels) / len(labels)
    accuracies.append(accuracy.item())
    print(f'batch: {batch + 1}/{len(train_loader)}, train loss: {loss.item()}, train accuracy: {accuracy.item()}, time: {round(perf_counter() - start_time, 1)}s')
  train_loss = np.mean(losses)
  train_accuracy = np.mean(accuracies)
  return train_loss, train_accuracy

def test(model, test_loader):
  print('testing...')
  model.eval()
  losses = []
  accuracies = []
  with torch.no_grad():
    for batch, (images, labels) in enumerate(test_loader):
      labels = labels.squeeze()
      pred = model(images)
      loss = F.cross_entropy(pred, labels)
      losses.append(loss.item())
      accuracy = torch.sum(torch.argmax(pred, 1) == labels) / len(labels)
      accuracies.append(accuracy.item())
      print(f'batch: {batch + 1}/{len(test_loader)}, test loss: {loss.item()}, test accuracy: {accuracy.item()}, time: {round(perf_counter() - start_time, 1)}s')
  test_loss = np.mean(losses)
  test_accuracy = np.mean(accuracies)
  return test_loss, test_accuracy

def main():
  images_df = pd.read_csv(IMAGES_CSV)
  train_size = int(0.9 * len(images_df))
  train_df = images_df[:train_size]
  test_df = images_df[train_size:]

  coords = train_df[['lat', 'lng']].to_numpy()
  gm = GaussianMixture(n_components=NUM_CLASSES, random_state=0).fit(coords)

  train_data = GeoData(train_df, gm)
  test_data = GeoData(test_df, gm)
  train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

  # clusters = gm.predict(coords)
  # counts = {}
  # for cluster in clusters:
  #   if not cluster in counts:
  #     counts[cluster] = 0
  #   counts[cluster] += 1
  # class_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
  # print(class_counts)
  # exit()

  if not os.path.isdir(TRAIN_OUT):
    os.makedirs(TRAIN_OUT)
    with open(LOSSES_CSV, 'w') as losses_csv:
      loss_writer = csv.writer(losses_csv)
      loss_writer.writerow(['train_loss', 'test_loss'])
    with open(ACCURACIES_CSV, 'w') as accuracies_csv:
      accuracy_writer = csv.writer(accuracies_csv)
      accuracy_writer.writerow(['train_accuray', 'test_accuracy'])

  epoch = 0
  train_losses = []
  test_losses = []
  with open(LOSSES_CSV, 'r') as losses_csv:
    next(losses_csv)
    for row in losses_csv:
      epoch += 1
      train_loss, test_loss = eval(row)
      train_losses.append(train_loss)
      test_losses.append(test_loss)
  
  train_accuracies = []
  test_accuracies = []
  with open(ACCURACIES_CSV, 'r') as accuracies_csv:
    next(accuracies_csv)
    for row in accuracies_csv:
      train_accuracy, test_accuracy = eval(row)
      train_accuracies.append(train_accuracy)
      test_accuracies.append(test_accuracy)

  if os.path.isfile(MODEL_PATH):
    model = torch.load(MODEL_PATH)
  else:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # only train final layer
    for param in model.parameters():
      param.requires_grad = False
    # re-initialize final fully-connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)

  with open(LOSSES_CSV, 'a') as losses_csv, open(ACCURACIES_CSV, 'a') as accuracies_csv:
    loss_writer = csv.writer(losses_csv)
    accuracy_writer = csv.writer(accuracies_csv)

    while epoch < NUM_EPOCHS:
      epochs_per_lr = NUM_EPOCHS // len(LEARNING_RATES)
      learning_rate = LEARNING_RATES[min(epoch // epochs_per_lr, len(LEARNING_RATES) - 1)]

      # optimizer only for training final fully-connected layer
      optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)

      train_loss, train_accuracy = train(model, train_loader, optimizer)
      test_loss, test_accuracy = test(model, test_loader)

      train_losses.append(train_loss)
      test_losses.append(test_loss)
      loss_writer.writerow([train_loss, test_loss])

      train_accuracies.append(train_accuracy)
      test_accuracies.append(test_accuracy)
      accuracy_writer.writerow([train_accuracy, test_accuracy])

      torch.save(model, MODEL_PATH)

      plt.figure()
      plt.title('Loss vs. Epoch')
      plt.plot(range(epoch + 1), train_losses, label='Train')
      plt.plot(range(epoch + 1), test_losses, label='Test')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.legend()
      plt.savefig(LOSSES_PLOT)

      plt.figure()
      plt.title('Accuracy vs. Epoch')
      plt.plot(range(epoch + 1), train_accuracies, label='Train')
      plt.plot(range(epoch + 1), test_accuracies, label='Test')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.legend()
      plt.savefig(ACCURACIES_PLOT)

      print(f'epoch: {epoch + 1}/{NUM_EPOCHS}, train accuracy: {train_accuracy}, test accuracy: {test_accuracy}, time: {round(perf_counter() - start_time, 1)}s')
      epoch += 1

  print(f'done!')

if __name__ == '__main__':
  main()
