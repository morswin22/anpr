# pylint: disable=maybe-no-member
import cv2 as cv
import numpy as np
import string
from PIL import ImageFont, ImageDraw, Image
from random import uniform, random, choices

padding = 50
width = 250
height = 100
angle = 40
letters = 5

font_size = 1

font = ImageFont.truetype('polish.ttf', font_size)
while font.getsize('W'*letters)[0] < width:
  font_size += 1
  font = ImageFont.truetype('polish.ttf', font_size)

text_offset = (height - font.getsize('A')[1]) / 2

while True:
  text = ''.join(choices(string.ascii_uppercase + string.digits, k=letters))

  src = np.zeros(shape=[height + 2*padding, width + 2*padding, 3], dtype=np.uint8)

  src_pil = Image.fromarray(src)
  draw = ImageDraw.Draw(src_pil)
  draw.rectangle([(padding, padding), (width+padding, height+padding)], fill=(255,255,255))
  draw.text((padding, padding + text_offset), text, font=font, fill=(0, 0, 0, 0))
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
  warp_rotate_dst = cv.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))

  cv.imshow('Number-plate', src)
  cv.imshow('Transformed', warp_rotate_dst)

  if cv.waitKey() == 27:
    break
