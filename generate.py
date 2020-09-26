import argparse
import json
import os
import string
from random import choice, choices, random, uniform

import cv2 as cv
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image, ImageDraw, ImageFont


def sp_noise(image, prob):
  output = np.zeros(image.shape, np.uint8)
  thres = 1 - prob 
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      rdn = random()
      if rdn < prob:
        output[i][j] = 0
      elif rdn > thres:
        output[i][j] = 255
      else:
        output[i][j] = image[i][j]
  return output

parser = argparse.ArgumentParser(description='Generate dataset for ANPR CNN')
parser.add_argument('-c', '--count', dest='count', action='store', default=1000, help='number of images for generate')
parser.add_argument('-d', '--dir', dest='dir', action='store', default='generated', help='directory name for storing the output')
arguments = parser.parse_args()

save_path = arguments.dir
generate = int(arguments.count)

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

if os.path.exists(save_path):
  print(f'Remove {save_path}/ to continue')
  exit()

os.makedirs(save_path)

ds = tfds.load("sun397/standard-part1-120k", split='train+test', shuffle_files=True)

with open('numbers.json', 'r') as file:
  numbers = json.load(file)

uppercase = string.ascii_uppercase
for letter in numbers['removed']: 
  uppercase = uppercase.replace(letter, '') 

font_size = 1
font = ImageFont.truetype('polish.ttf', font_size)
while font.getsize(' '+'E'*letters)[0] < width - aside:
  font_size += 1
  font = ImageFont.truetype('polish.ttf', font_size)
text_offset = -5 + (height - font.getsize('A')[1]) / 2

path = save_path + '/'
for i, example in enumerate(ds.take(generate)):
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

  grayscale = cv.cvtColor(sp_noise(bg, 0.02), cv.COLOR_BGR2GRAY)
  cv.imwrite(path + str(i) + '_' + is_plate + text + '.png', grayscale)
