import json
import os
import re
from datetime import datetime
import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.utils import shuffle
from tensorflow.keras import layers, models
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train the CNN model')
parser.add_argument('-s', '--splits', dest='splits', action='store', default='1', help='number of splits')
parser.add_argument('-e', '--epochs', dest='epochs', action='store', default='10', help='number of epochs')
parser.add_argument('-t', '--train', dest='train', action='store', default='0.8', help='percentage of training data in dataset')
parser.add_argument('-d', '--dir', dest='dir', action='store', default='generated-1-10-397000', help='dataset path')
parser.add_argument('-m', '--model', dest='model', action='store', default=None, help='path of the model to use')

with open('assets/map.json', 'r') as file:
  mapped = json.load(file)

def encoder(label):
  result = np.zeros((202))
  if label != '0_no_text':
    offset = 0
    for i, letter in enumerate(label):
      if i != 0 or letter != '0':
        result[offset + mapped[i].index(letter)] = 1
      offset += len(mapped[i])
  return result

def decoder(output):
  label = '0' if output[0] < 0.5 else '1'
  offset = 1
  for letters in mapped[1:]:
    index = np.argmax(output[offset:offset+len(letters)])
    label += letters[index]
    offset += len(letters)
  return label

arguments = parser.parse_args()

dataset_path, train_with, n_splits, n_epochs = arguments.dir, float(arguments.train), max(int(arguments.splits), 1), int(arguments.epochs)

if not os.path.isdir(dataset_path):
  print(f'Dataset was not found at {dataset_path}')
  exit()
dataset_files = list(os.scandir(dataset_path))

if arguments.model is None:
  model = models.Sequential()
  model.add(layers.Conv2D(48, (5, 5), activation='relu', input_shape=(128, 64, 1)))
  model.add(layers.MaxPool2D((2, 2)))
  model.add(layers.Conv2D(64, (5, 5), activation='relu'))
  model.add(layers.MaxPool2D((1, 2)))
  model.add(layers.Conv2D(128, (5, 5), activation='relu'))
  model.add(layers.MaxPool2D((2, 2)))
  model.add(layers.Flatten())
  model.add(layers.Dense(2048, activation='relu'))
  model.add(layers.Dense(202, activation='sigmoid'))
else:
  model = models.load_model(arguments.model)
model.summary()

history = { 'loss': [], 'accuracy': [], 'val_accuracy': [], 'acc': [] }

for i in range(n_splits):
  files = dataset_files[i::n_splits]
  dataset_length = len(files)
  split = int(dataset_length * train_with)

  pattern = re.compile("\d+_(.+)\.png")
  ds_data = []
  ds_labels = []
  for path in tqdm(files, unit='example', total=dataset_length):
    result = pattern.search(path.name)
    label = encoder(result.group(1))
    ds_labels.append(label)
    image = np.array(Image.open(os.path.join(dataset_path, path.name)), dtype=np.uint8).reshape((128, 64, 1)) / 255
    ds_data.append(image)

  shuffle(ds_data, ds_labels)
  (train_images, train_labels), (test_images, test_labels) = ((np.array(ds_data[:split]), np.array(ds_labels[:split])), (np.array(ds_data[split:]), np.array(ds_labels[split:])))

  model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

  result = model.fit(train_images, train_labels, epochs=n_epochs, validation_data=(test_images, test_labels))
  history['loss'] += result.history['loss']
  history['accuracy'] += result.history['accuracy']
  history['val_accuracy'] += result.history['val_accuracy']

  test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
  history['acc'].append(test_acc)

plt.plot(history['loss'], label='loss')
plt.plot(history['accuracy'], label='accuracy')
plt.plot(history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

print('Accuracy across splits')
for i, accuracy in enumerate(history['acc']):
  print(f'#{i+1} accuracy {accuracy}')

os.makedirs('models', exist_ok=True)
name = f'models/{str(datetime.now())} model.h5'.replace(':','-').replace(' ','_')
model.save(name)

image = np.array(Image.open('assets/test.png'), dtype=np.uint8).reshape((128, 64, 1)) / 255
predictions = model.predict(np.array([image]))
print('Ground truth: 1WE 295GC')
print(f'Prediction: {decoder(predictions[0])}')

plt.show()
