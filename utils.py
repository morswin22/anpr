import json

import numpy as np

with open('assets/map.json', 'r') as file:
  mapped = json.load(file)

def encoder(label):
  result = np.zeros((207))
  offset = 0
  for i, letter in enumerate(label):
    result[offset + mapped[i].index(letter)] = 1
    offset += len(mapped[i])
  return result

def decoder(output):
  label = ''
  offset = 0
  for letters in mapped:
    index = np.argmax(output[offset:offset+len(letters)])
    label += letters[index]
    offset += len(letters)
  return label
