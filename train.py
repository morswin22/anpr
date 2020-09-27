import json
import os
import re
import shutil
from subprocess import Popen

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models

with open('map.json', 'r') as file:
  mapped = json.load(file)

def map_label(label):
  result = np.zeros((202))
  if label != '0_no_text':
    offset = 0
    for i, letter in enumerate(label):
      if i != 0 or letter != '0':
        result[offset + mapped[i].index(letter)] = 1
      offset += len(mapped[i])
  return result

def map_output(output):
  label = ''
  offset = 0
  for letters in mapped:
    index = np.argmax(output[offset:offset+len(letters)])
    label += letters[index]
    offset += len(letters)
  return label

generate_count = 30000
generate_path = 'generated'
train_with = .85

Popen(f'python generate.py -c {generate_count} -d {generate_path}').wait()

split = int(generate_count * train_with)
pattern = re.compile("\d+_(.+)\.png")
ds_data = []
ds_labels = []
for path in os.scandir(generate_path):
  result = pattern.search(path.name)
  label = map_label(result.group(1))
  ds_labels.append(label)
  image = np.array(Image.open(os.path.join(generate_path, path.name)), dtype=np.uint8).reshape((128, 64, 1)) / 255
  ds_data.append(image)
ds = ((np.array(ds_data[:split]), np.array(ds_labels[:split])), (np.array(ds_data[split:]), np.array(ds_labels[split:])))

# shutil.rmtree('generated')

(train_images, train_labels), (test_images, test_labels) = ds 

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

plt.plot(history.history['binary_crossentropy'], label='accuracy')
plt.plot(history.history['val_binary_crossentropy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.1, 1])
plt.legend(loc='lower right')

plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(test_acc)

image = np.array(Image.open('test.png'), dtype=np.uint8).reshape((128, 64, 1)) / 255
predictions = model.predict(np.array([image]))
print('Ground truth: 1WE 295GC')
print(f'Prediction: {map_output(predictions[0])}')