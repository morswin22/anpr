import argparse
import json
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
padding = 70, 120
delta = 50
width = 520
height = 114
angle = 12
no_plate = 0.02
# TODO Add varying number of letters
letters = 7
aside = 45
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

with open('assets/numbers.json', 'r') as file:
  numbers = json.load(file)

uppercase = string.ascii_uppercase
for letter in numbers['removed']: 
  uppercase = uppercase.replace(letter, '') 

font_size = 1
font = ImageFont.truetype('assets/polish.ttf', font_size)
while font.getsize(' '+'E'*letters)[0] < width - aside:
  font_size += 1
  font = ImageFont.truetype('assets/polish.ttf', font_size)
text_offset = -5 + (height - font.getsize('A')[1]) / 2

def generate(part, count):
  ds = tfds.load(f"sun397/standard-part{part}-120k", split='train+test', shuffle_files=True, data_dir=arguments.set)

  path = save_path + '/' + str(part) + str(count)
  for i, example in tqdm(enumerate(ds.take(count)), unit="example", total=count):
    background = example['image']

    if random() > no_plate:
      county = choice(numbers['counties'])
      text = county + (' ' if len(county) < 3 else '') + ''.join(choices(uppercase + string.digits, k=letters-len(county)))

      src = np.zeros(shape=[height + 2*padding[1], width + 2*padding[0], 4], dtype=np.uint8)

      src_pil = Image.fromarray(src)
      draw = ImageDraw.Draw(src_pil)
      draw.rectangle([padding, (width+padding[0], height+padding[1])], fill=(255, 255, 255, 255))
      draw.text((padding[0] + aside, padding[1] + text_offset), text, font=font, fill=(0, 0, 0, 255))
      src = np.array(src_pil)

      srcTri = np.array( [[0, 0], 
                          [src.shape[1] - 1, 0], 
                          [0, src.shape[0] - 1]] ).astype(np.float32)

      dstTri = np.array( [[random()*delta, random()*delta],
                          [src.shape[1]-random()*delta, random()*delta], 
                          [random()*delta, src.shape[0]-random()*delta]] ).astype(np.float32)

      warp_mat = cv.getAffineTransform(srcTri, dstTri)
      warp_dst = cv.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))

      rot_mat = cv.getRotationMatrix2D( (warp_dst.shape[1]//2, warp_dst.shape[0]//2), uniform(-angle, angle), 1 )
      transformed = cv.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))

      xy = np.where(transformed==transformed.max())
      y = xy[0][0], xy[0][len(xy[0])-1]
      x = xy[1][0], xy[1][len(xy[0])-1]
      w = x[1] - x[0]
      h = y[1] - y[0]
      is_plate = '0' if 0.6 <= w / transformed.shape[1] <= 0.8 or .6 <= h / transformed.shape[0] <= .875 or y[0] == 0 or y[1] == transformed.shape[0] -1 or x[0] == 0 or x[1] == transformed.shape[1] - 1 else '1'

      transformed_pil = Image.fromarray(transformed)
      ratio = background.shape[1] / background.shape[0]
      bg_pil = Image.fromarray(np.array(background[...,::-1])).resize((transformed.shape[1], int(transformed.shape[1] / min(ratio, 2))))
      bg_pil.paste(transformed_pil, box=(0,0), mask=transformed_pil)
      bg = np.array(bg_pil.crop(box=(0, 0, transformed.shape[1], transformed.shape[1] // 2)).resize(final_size))
    else:
      bg = np.array(Image.fromarray(np.array(background)).resize(final_size))
      text = '_no_text'
      is_plate = '0'

    grayscale = cv.cvtColor(np.clip(bg + np.random.normal(0, 5, np.prod(bg.shape)).reshape(bg.shape), 0, 255).astype('uint8'), cv.COLOR_BGR2GRAY)
    cv.imwrite(path + str(i) + '_' + is_plate + text + '.png', grayscale)

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
