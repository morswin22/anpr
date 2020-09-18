import cv2 as cv
import numpy as np
import string
from random import uniform, random, choices

diff = 30
angle = 30
letters = 5

while True:
  text = ''.join(choices(string.ascii_uppercase + string.digits, k=letters))

  src = 255 * np.ones(shape=[75, 150, 3], dtype=np.uint8)

  cv.putText(src, text, (0, 55), cv.FONT_HERSHEY_PLAIN, min(src.shape[0],src.shape[1])/25, 0, thickness=3);

  srcTri = np.array( [[0, 0], [src.shape[1] - 1, 0], [0, src.shape[0] - 1]] ).astype(np.float32)
  # TODO Fix dstTri sometimes overflowing boundaries
  dstTri = np.array( [[random()*diff, random()*diff], [src.shape[1]-random()*diff, random()*diff], [random()*diff, src.shape[0]-random()*diff]] ).astype(np.float32)

  warp_mat = cv.getAffineTransform(srcTri, dstTri)
  warp_dst = cv.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))

  rot_mat = cv.getRotationMatrix2D( (warp_dst.shape[1]//2, warp_dst.shape[0]//2), uniform(-angle, angle), 1 )
  warp_rotate_dst = cv.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))

  cv.imshow('Source image', src)
  cv.imshow('Warp', warp_dst)
  cv.imshow('Warp + Rotate', warp_rotate_dst)

  if cv.waitKey() == 27:
    break
