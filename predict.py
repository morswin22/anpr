import argparse
import json
import os
import re
from random import choice

import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras import models

parser = argparse.ArgumentParser(description='Predict number-plate using the CNN model')
parser.add_argument('-i', '--image', dest='image', action='store', default='assets/test.png', help='input image path')
parser.add_argument('-m', '--model', dest='model', action='store', default='models/anpr-300k.h5', help='trained model path')
parser.add_argument('-r', '--random', dest='random', action='store', default=None, help='use random examples from given dataset')

arguments = parser.parse_args()

if os.path.isfile(arguments.model):
  model = models.load_model(arguments.model)

  with open('assets/map.json', 'r') as file:
    mapped = json.load(file)

  def decoder(output):
    label = ''
    offset = 0
    for letters in mapped:
      index = np.argmax(output[offset:offset+len(letters)])
      label += letters[index]
      offset += len(letters)
    return label

  if arguments.random:
    if not os.path.isdir(arguments.random):
      print(f'Dataset was not found at {arguments.random}')
      exit()
    files = list(os.scandir(arguments.random))
    pattern = re.compile("\d+_(.+)\.png")

    while True:
      file = choice(files)

      mat = cv.imread(file.path, cv.IMREAD_GRAYSCALE)
      image = np.array(Image.open(file.path), dtype=np.uint8).reshape((128, 64, 1)) / 255

      label = pattern.search(file.name).group(1)
      predictions = model.predict(np.array([image]))

      cv.imshow('Input', mat)
      print('Ground truth vs prediction')
      print(label)
      print(decoder(predictions[0]))

      if cv.waitKey() == 27:
        break
  else:
    if os.path.isfile(arguments.image):
      image_pil = ImageOps.grayscale(Image.open(arguments.image).resize((128, 64)))
      image = np.array(image_pil, dtype=np.uint8).reshape((128, 64, 1)) / 255

      predictions = model.predict(np.array([image]))

      print(decoder(predictions[0]))
    else:
      print(f'Input image was not found at {arguments.image}')
else:
  print(f'Model was not found at {arguments.model}')