import json
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.utils import shuffle
from tensorflow.keras import layers, models
from tqdm import tqdm

with open('map.json', 'r') as file:
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

dataset_path = 'generated'
train_with = .85

dataset_files = list(os.scandir(dataset_path))
dataset_length = len(dataset_files)
split = int(dataset_length * train_with)

pattern = re.compile("\d+_(.+)\.png")
ds_data = []
ds_labels = []
for path in tqdm(dataset_files, unit='example', total=dataset_length):
  result = pattern.search(path.name)
  label = encoder(result.group(1))
  ds_labels.append(label)
  image = np.array(Image.open(os.path.join(dataset_path, path.name)), dtype=np.uint8).reshape((128, 64, 1)) / 255
  ds_data.append(image)

shuffle(ds_data, ds_labels)
(train_images, train_labels), (test_images, test_labels) = ((np.array(ds_data[:split]), np.array(ds_labels[:split])), (np.array(ds_data[split:]), np.array(ds_labels[split:]))) 

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

model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryCrossentropy()])

history = model.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels))

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['binary_crossentropy'], label='binary_crossentropy')
plt.plot(history.history['val_binary_crossentropy'], label='val_binary_crossentropy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(1 - test_acc)

os.makedirs('models', exist_ok=True)
name = f'models/{str(datetime.now())} model.h5'.replace(':','-').replace(' ','_')
model.save(name)

image = np.array(Image.open('test.png'), dtype=np.uint8).reshape((128, 64, 1)) / 255
predictions = model.predict(np.array([image]))
print('Ground truth: 1WE 295GC')
print(f'Prediction: {decoder(predictions[0])}')

plt.show()
