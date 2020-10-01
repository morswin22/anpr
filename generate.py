import argparse
import json
import math
import os
import shutil
import string
import time
from multiprocessing import Process
from random import choice, choices, random, uniform
from threading import Thread

import cv2 as cv
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Generate dataset for ANPR CNN')
parser.add_argument('-c', '--count', dest='count', action='store', default='30000', help='number of images for generate')
parser.add_argument('-p', '--part', dest='part', action='store', default='-1', help='index of SUN397 part')
parser.add_argument('-d', '--dir', dest='dir', action='store', default='generated-{part}-{count}', help='directory pattern for storing the output')
parser.add_argument('-m', '--mode', dest='mode', action='store', default='0', help='0 - single thread, 1 - multithread, 2 - multiprocess')
parser.add_argument('-s', '--set', dest='set', action='store', default='D:\\tensorflow_datasets', help='directory of downloaded tensorflow datasets')
parser.add_argument('-f', '--force', dest='force', action='store_true', help='flag to overwrite directory')
parser.add_argument('-a', '--all', dest='all', action='store_true', help='flag to use all avaible backgrounds')

MAX_IN_PART = 39700
MAX_IN_SET = MAX_IN_PART * 10
MIN_PART, MAX_PART = 1, 10
width = 200
aside = int(0.088 * width)
final_size = 128, 64

arguments = parser.parse_args()

count = MAX_IN_SET if arguments.all else min(int(arguments.count), MAX_IN_SET)
part = int(arguments.part)

parts = None
if count > MAX_IN_PART and part < 0:
  remainder = count % MAX_IN_PART
  parts = [MAX_IN_PART for i in range(count // MAX_IN_PART)]
  if remainder > 0:
    parts.append(remainder)
  part = f'1-{len(parts)}'
else:
  part = str(max(min(abs(int(arguments.part)), MAX_PART), MIN_PART))
save_path = arguments.dir.replace('{part}', part).replace('{count}', str(count))

if os.path.exists(save_path):
  if arguments.force:
    shutil.rmtree(save_path)
  else:
    print(f'Remove {save_path}/ to continue')
    exit()
os.makedirs(save_path)

with open('assets/map.json', 'r') as file:
  mapped = json.load(file)

font_sizes = {8: None, 7: None, 6: None}
font_size = 1
for letters in font_sizes:
  font = ImageFont.truetype('assets/polish.ttf', font_size)
  example = ' ' + 'E'*letters
  while font.getsize(example)[0] < width - aside:
    font_size += 1
    font = ImageFont.truetype('assets/polish.ttf', font_size)
  text_width, font_height = font.getsize(example)
  font_sizes[letters] = font, font_height, text_width

text_probability = 0.4
text_probabilities = [text_probability] + [(1-text_probability)/15] * 15, [text_probability] + [(1-text_probability)/30] * 30

def euler_to_mat(yaw, pitch, roll):
  # Rotate clockwise about the Y-axis
  c, s = math.cos(yaw), math.sin(yaw)
  M = np.matrix([[ c, 0., s], [ 0., 1., 0.], [ -s, 0., c]])

  # Rotate clockwise about the X-axis
  c, s = math.cos(pitch), math.sin(pitch)
  M = np.matrix([[ 1., 0., 0.], [ 0., c, -s], [ 0., s, c]]) * M

  # Rotate clockwise about the Z-axis
  c, s = math.cos(roll), math.sin(roll)
  M = np.matrix([[ c, -s, 0.], [ s, c, 0.], [ 0., 0., 1.]]) * M

  return M

def generate(part, count):
  ds = tfds.load(f"sun397/standard-part{part}-120k", split='train+test', shuffle_files=True, data_dir=arguments.set)

  path = save_path + '/' + str(part) + str(count)
  for i, example in tqdm(enumerate(ds.take(count)), unit="examples", total=count):
    # Extract background
    background = np.array(example['image'][...,::-1], np.uint8)

    # Generate text
    county = ''.join([choice(chars) for chars in mapped[2][:-1]] + [np.random.choice(chars, p=text_probabilities[0]) for chars in mapped[2][-1:]])
    vehicle = ''.join([choice(chars) for chars in mapped[3][:-1]] + [np.random.choice(chars, p=text_probabilities[1]) for chars in mapped[3][-1:]])
    text = county.strip() + ' ' + vehicle.strip()
    letters = len(text) - 1
    font, font_height, text_width = font_sizes[letters]

    h_padding = uniform(0.2, 0.4) * font_height
    v_padding = uniform(0.1, 0.3) * font_height
    radius = 1 + int(font_height * 0.1 * random())
    out_shape = int(font_height + v_padding * 2), int(text_width + aside + h_padding * 2)

    # Generate plate mask
    plate_mask = np.ones([*out_shape, 3])
    plate_mask[:radius, :radius] = [0.0] * 3
    plate_mask[-radius:, :radius] = [0.0] * 3
    plate_mask[:radius, -radius:] = [0.0] * 3
    plate_mask[-radius:, -radius:] = [0.0] * 3

    cv.circle(plate_mask, (radius, radius), radius, [1.0] * 3, -1)
    cv.circle(plate_mask, (radius, out_shape[0] - radius), radius, [1.0] * 3, -1)
    cv.circle(plate_mask, (out_shape[1] - radius, radius), radius, [1.0] * 3, -1)
    cv.circle(plate_mask, (out_shape[1] - radius, out_shape[0] - radius), radius, [1.0] * 3, -1)

    # Pick colors
    first = True
    while first or plate_color - text_color < 0.3:
        text_color = random()
        plate_color = random()
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    plate_color = tuple([int(plate_color * 255)] * 3)
    text_color = tuple([int(text_color * 255)] * 3)

    # Generate plate
    text_pil = Image.fromarray(np.zeros([*out_shape, 3], dtype=np.uint8))
    draw = ImageDraw.Draw(text_pil)
    draw.rectangle(((0, 0), out_shape[::-1]), fill=plate_color)
    draw.text((h_padding + aside, v_padding - 2), text, font=font, fill=text_color)
    plate = np.array(text_pil)

    # Transform
    out_of_bounds = False
    from_size = np.array([[plate.shape[1], plate.shape[0]]]).T
    to_size = np.array([[background.shape[1], background.shape[0]]]).T

    scale_variation = 1.5
    rotation_variation = 1.0
    translation_variation = 1.2

    min_scale, max_scale = 0.6, 0.875
    scale = uniform((min_scale + max_scale) * 0.5 - (max_scale - min_scale) * 0.5 * scale_variation, (min_scale + max_scale) * 0.5 + (max_scale - min_scale) * 0.5 * scale_variation)

    if scale > max_scale or scale < min_scale:
      out_of_bounds = True
    
    roll = uniform(-0.3, 0.3) * rotation_variation
    pitch = uniform(-0.2, 0.2) * rotation_variation
    yaw = uniform(-1.2, 1.2) * rotation_variation

    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w, d = plate.shape
    corners = np.matrix([[-w, w, -w, w], [-h, -h, h, h]]) * 0.5
    skewed_size = np.array(np.max(M * corners, axis=1) - np.min(M * corners, axis=1))

    scale *= np.min(to_size / skewed_size)

    trans = (np.random.random((2, 1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if np.any(trans < -0.5) or np.any(trans > 0.5):
      out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.0
    center_from = from_size / 2.0

    M *= scale
    M = np.hstack([M, trans + center_to - M * center_from])

    plate = cv.warpAffine(plate, M, (background.shape[1], background.shape[0]))
    plate_mask = cv.warpAffine(plate_mask, M, (background.shape[1], background.shape[0]))

    # Merge plate with background, turn grayscale and add noise
    out = cv.cvtColor(cv.resize(np.array(plate * plate_mask + background * (1.0 - plate_mask), np.uint8), final_size), cv.COLOR_BGR2GRAY)
    out = np.clip(out + np.random.normal(0, 13, out.shape), 0, 255).astype('uint8')

    # Save
    cv.imwrite(path + str(i) + '_' + str(int(not out_of_bounds)) + str(len(county.strip())) + str(len(vehicle.strip())) + text + '.png', out)

if __name__ == "__main__":
  if parts is None:
    generate(part, count)
  else:
    if arguments.mode == '0':
      for i, count in enumerate(parts):
        generate(i + 1, count)
    elif arguments.mode == '1':
      threads = []
      for i, count in enumerate(parts):
        thread = Thread(target=generate, args=(i + 1, count), daemon=True)
        thread.start()
        threads.append(thread)
      while True in [thread.is_alive() for thread in threads]:
        time.sleep(1)
    elif arguments.mode == '2':
      processes = []
      for i, count in enumerate(parts):
        process = Process(target=generate, args=(i + 1, count), daemon=True)
        process.start()
        processes.append(process)
      try:
        while True in [process.is_alive() for process in processes]:
          time.sleep(1)
      except KeyboardInterrupt:
        for process in processes:
          process.terminate()
