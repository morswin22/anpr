import argparse
import os
import pathlib
import re
from datetime import datetime
from random import shuffle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras import callbacks, layers, models
from tqdm import tqdm, trange

from utils import decoder, encoder

parser = argparse.ArgumentParser(description='Train the CNN model')
parser.add_argument('-s', '--splits', dest='splits', action='store', default='1', help='number of splits')
parser.add_argument('-e', '--epochs', dest='epochs', action='store', default='10', help='number of epochs')
parser.add_argument('-t', '--train', dest='train', action='store', default='0.8', help='percentage of training data in dataset')
parser.add_argument('-d', '--dir', dest='dir', action='store', default='generated-1-10-397000', help='dataset path')
parser.add_argument('-m', '--model', dest='model', action='store', default=None, help='path of the model to use')
parser.add_argument('-c', '--checkpoint', dest='checkpoint', action='store', default=None, help='path of the weights checkpoint to use')

arguments = parser.parse_args()

dataset_path, train_with, n_splits, n_epochs = arguments.dir, float(arguments.train), max(int(arguments.splits), 1), int(arguments.epochs)

if not os.path.isdir(dataset_path):
  print(f'Dataset was not found at {dataset_path}')
  exit()
dataset_files = list(os.scandir(dataset_path))
shuffle(dataset_files)
dataset_length = len(dataset_files) // n_splits
split = int(dataset_length * train_with)
pattern = re.compile("\d+_(.+)\.png")

if arguments.model is None:
  model = models.Sequential()

  model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 64, 1)))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPool2D((2, 2)))
  model.add(layers.Dropout(0.2))

  model.add(layers.Conv2D(128, (3, 3), activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPool2D((2, 1)))
  model.add(layers.Dropout(0.2))

  model.add(layers.Conv2D(256, (3, 3), activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPool2D((2, 2)))
  model.add(layers.Dropout(0.2))

  model.add(layers.Conv2D(512, (3, 3), activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPool2D((2, 2)))
  model.add(layers.Dropout(0.2))

  model.add(layers.Flatten())
  model.add(layers.Dense(4096, activation='relu'))
  model.add(layers.Dropout(0.3))

  model.add(layers.Dense(207, activation='sigmoid'))
else:
  model = models.load_model(arguments.model)
model.summary()

if arguments.checkpoint:
  if os.path.isfile(arguments.checkpoint + 'model.index'):
    model.load_weights(arguments.checkpoint + 'model')
  else:
    print(f'Checkpoint was not found at {arguments.checkpoint}model')
    exit()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
tensorboard = callbacks.TensorBoard(os.path.join(pathlib.Path().absolute(), 'logs', name), histogram_freq=1)
checkpoint = callbacks.ModelCheckpoint(os.path.join(pathlib.Path().absolute(), 'checkpoints', name, 'model'), save_best_only=True, save_weights_only=True)

for i in trange(n_splits, total=n_splits, desc='Splits', unit='split'):
  train_images, train_labels, test_images, test_labels, result = [None] * 5
  ds_data = []
  ds_labels = []
  
  files = dataset_files[i::n_splits]
  for file in tqdm(files, unit='examples', total=dataset_length):
    ds_labels.append(encoder(pattern.search(file.name).group(1)))
    ds_data.append(np.array(Image.open(file.path), dtype=np.uint8).reshape((128, 64, 1)) / 255)

  files = None
  (train_images, train_labels), (test_images, test_labels) = ((np.array(ds_data[:split]), np.array(ds_labels[:split])), (np.array(ds_data[split:]), np.array(ds_labels[split:])))
  ds_data, ds_labels = None, None

  model.fit(train_images, train_labels, epochs=n_epochs, validation_data=(test_images, test_labels), callbacks=[tensorboard, checkpoint])
  model.evaluate(test_images, test_labels)

os.makedirs('models', exist_ok=True)
model.save(f"models/{name}_model.h5")

image = np.array(ImageOps.grayscale(Image.open('assets/test.png').resize((128, 64))), dtype=np.uint8).reshape((128, 64, 1)) / 255
predictions = model.predict(np.array([image]))
print('Ground truth: 125PO 5HG85')
print(f'Prediction: {decoder(predictions[0])}')
