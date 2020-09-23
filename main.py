# pylint: disable=maybe-no-member
import cv2 as cv
import numpy as np
import string
from PIL import ImageFont, ImageDraw, Image
from random import uniform, random, choices, choice
import json
import tensorflow_datasets as tfds

n_backgrounds = 100
padding = 50
width = 520
height = 114
angle = 40
letters = 7
aside = 45

ds, ds_info = tfds.load("sun397/standard-part1-120k", split='train', shuffle_files=True, with_info=True)
tfds.show_examples(ds, ds_info)

backgrounds = []
for example in ds.take(n_backgrounds):
  backgrounds.append(example['image'])

font_size = 1

font = ImageFont.truetype('polish.ttf', font_size)
while font.getsize(' '+'E'*letters)[0] < width - aside:
  font_size += 1
  font = ImageFont.truetype('polish.ttf', font_size)

text_offset = -5 + (height - font.getsize('A')[1]) / 2

with open('numbers.json', 'r') as file:
  numbers = json.load(file)

uppercase = string.ascii_uppercase
for letter in numbers['removed']: 
  uppercase = uppercase.replace(letter, '') 

for background in backgrounds:
  county = choice(numbers['counties'])
  text = county + ' ' + ''.join(choices(uppercase + string.digits, k=letters-len(county)))

  src = np.zeros(shape=[height + 2*padding, width + 2*padding, 4], dtype=np.uint8)

  src_pil = Image.fromarray(src)
  draw = ImageDraw.Draw(src_pil)
  draw.rectangle([(padding, padding), (width+padding, height+padding)], fill=(255, 255, 255, 255))
  draw.text((padding + aside, padding + text_offset), text, font=font, fill=(0, 0, 0, 255))
  src = np.array(src_pil)

  srcTri = np.array( [[0, 0], 
                      [src.shape[1] - 1, 0], 
                      [0, src.shape[0] - 1]] ).astype(np.float32)

  dstTri = np.array( [[random()*padding, random()*padding],
                      [src.shape[1]-random()*padding, random()*padding], 
                      [random()*padding, src.shape[0]-random()*padding]] ).astype(np.float32)

  warp_mat = cv.getAffineTransform(srcTri, dstTri)
  warp_dst = cv.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))

  rot_mat = cv.getRotationMatrix2D( (warp_dst.shape[1]//2, warp_dst.shape[0]//2), uniform(-angle, angle), 1 )
  transformed = cv.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))

  transformed_pil = Image.fromarray(transformed)
  bg_pil = Image.fromarray(np.array(background))
  bg_pil = bg_pil.resize((transformed.shape[1], transformed.shape[0]))
  bg_pil.paste(transformed_pil, box=(0,0), mask=transformed_pil)
  bg = np.array(bg_pil)

  cv.imshow('Number-plate', src)
  cv.imshow('Transformed', transformed)
  cv.imshow('Final', bg)

  if cv.waitKey() == 27:
    break
