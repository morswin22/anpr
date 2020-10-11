import os
import tkinter as tk
from collections import Counter
from tkinter.filedialog import askopenfilename

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras import models

from utils import decoder

IMAGE_FORMATS = [("JPEG", "*.jpg"), ("PNG", "*.png")]
SIZE = 512
ZOOM_MULT = 2 ** (1 / 2)
FINAL_SHAPE = 128, 64, 1
SLIDE_STEP = 8 / FINAL_SHAPE[0]
RATIO = FINAL_SHAPE[0] / FINAL_SHAPE[1]
MODEL_PATH = 'models/anpr.h5'
  
def get_possible_label(current, next_chars, labels=[]):
  if len(next_chars) > 0:
    for char in next_chars[0]:
      get_possible_label(current+char, next_chars[1:], labels)
    return labels
  else:
    labels.append(current)
    
class App:
  def __init__(self, master):
    self.master = master
    master.title("Automatic Number-Plate Recogintion")

    if self.load_model():
      self.image_text = tk.StringVar()
      self.image_name = tk.Label(master, textvariable=self.image_text)
      self.image_name.pack()

      self.image_label = tk.Label(master)
      self.image_label.pack()

      self.prediction_text = tk.StringVar()
      self.prediction_label = tk.Label(master, textvariable=self.prediction_text)
      self.prediction_label.pack()

      self.load_button = tk.Button(master, text="Load image", command=self.load_image)
      self.load_button.pack()

      self.load_button = tk.Button(master, text="Rotate", command=self.rotate_image)
      self.load_button.pack()

      self.zoom_slider = tk.Scale(master, from_=1, to=10, orient=tk.HORIZONTAL)
      self.zoom_slider.pack()
      self.zoom_slider.set(5)

      self.load_button = tk.Button(master, text="Predict", command=self.predict)
      self.load_button.pack()

  def load_model(self):
    if os.path.isfile(MODEL_PATH):
      self.model = models.load_model(MODEL_PATH)
    else:
      self.error_label = tk.Label(self.master, text="Model was not found")
      self.error_label.pack()
      return False
    return True

  def load_image(self):
    path = askopenfilename(filetypes=IMAGE_FORMATS)
    self.image_text.set(os.path.split(path)[1])
    self.image = Image.open(path)
    self.display(self.image)

  def rotate_image(self):
    if not 'image' in self.__dict__:
      return
    self.image = self.image.rotate(-90, expand=True)
    self.display(self.image)

  def display(self, image):
    w, h = image.size
    self.display_w, self.display_h = (SIZE, h * SIZE // w) if w > h else (w * SIZE // h, SIZE)
    self.display_image = ImageTk.PhotoImage(image.resize((self.display_w, self.display_h), Image.ANTIALIAS))
    self.image_label.configure(image=self.display_image)
    self.prediction_text.set('')

  def predict(self):
    if not 'image' in self.__dict__:
      return
    cropped, bboxes, valid, valid_bboxes, groups, max_probs, labels = [], [], [], [], [], [], []
    image = np.array(self.image.convert('L'), np.uint8)
    h, w = image.shape
    if w / h < RATIO:
      width = w
      height = width / RATIO
    else:
      height = h
      width = height * RATIO
    width, height = int(width), int(height)
  
    zoom = 1
    m_zoom = 2 ** (self.zoom_slider.get() / 2)
    while zoom <= m_zoom:
      scaled_w, scaled_h = int(w * zoom), int(h * zoom)
      
      overflow_x, overflow_y = abs(width - scaled_w), abs(height - scaled_h)
      coeff = w / scaled_w

      scaled = cv.resize(image, (scaled_w, scaled_h))

      step = int(SLIDE_STEP * scaled_w / 2)
      for i in range(0, overflow_x + step, step):
        for j in range(0, overflow_y + step, step):
          bboxes.append(((int(i*coeff), int(j*coeff)), (int(i*coeff + width*coeff), int(j*coeff + height*coeff))))
          cropped.append(cv.resize(scaled[j:j+height, i:i+width], FINAL_SHAPE[:-1]).reshape(FINAL_SHAPE) / 255)

      zoom *= ZOOM_MULT

    predictions = self.model.predict(np.array(cropped))

    img = np.array(self.image)
    for i, prediction in enumerate(predictions):
      code = decoder(prediction)
      if code[:1] == '1':
        valid.append(code[3:])
        valid_bboxes.append(bboxes[i])
        # img = cv.rectangle(img, *bboxes[i], (0, 255, 0), thickness=int(self.display_w*0.02))

    for i, bbox0 in enumerate(valid_bboxes):
      for j, bbox1 in enumerate(valid_bboxes[i+1:]):
        are_overlapping = max(bbox0[0][0], bbox1[0][0]) < min(bbox0[1][0], bbox1[1][0]) and max(bbox0[0][1], bbox1[0][1]) < min(bbox0[1][1], bbox1[1][1])
        if are_overlapping:
          appended = False
          for group in groups:
            if i in group or j+i+1 in group:
              if not i in group:
                group.append(i)
              if not j+i+1 in group:
                group.append(j+i+1)
              appended = True
          if not appended:
            groups.append([i, j+i+1])

    for i, bbox in enumerate(valid_bboxes):
      is_in_group = False
      for group in groups:
        if i in group:
          is_in_group = True
          break
      if not is_in_group:
        groups.append([i])

    for group in groups:
      top, bottom, left, right, length = 0, 0, 0, 0, len(group)
      letters = [[], [], [], [], [], [], [], []]
      for index in group:
        left += valid_bboxes[index][0][0]
        top += valid_bboxes[index][0][1]
        right += valid_bboxes[index][1][0]
        bottom += valid_bboxes[index][1][1]
        for i, letter in enumerate(valid[index]):
          letters[i].append(letter)

      for letter in letters:
        c = Counter(letter)
        max_prob = c.most_common(1)[0][1]
        with_max_prob = []
        for pair in c.most_common():
          if pair[1] == max_prob:
            with_max_prob.append(pair[0])
          elif pair[1] < max_prob:
            break
        max_probs.append(with_max_prob)
      
      possible = get_possible_label('', max_probs)
      if len(possible) > length:
        print('Unsure about group with labels: '+ ', '.join(possible))
      else:
        label = '/'.join(possible)
        labels.append(label)
        img = cv.putText(img, label, (left//length + int(self.display_w*0.1), top//length + int(self.display_w*0.3)), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), thickness=int(self.display_w*0.06))
        img = cv.rectangle(img, (left//length, top//length), (right//length, bottom//length), (0, 255, 0), thickness=int(self.display_w*0.02))

    self.display(Image.fromarray(img))
    self.prediction_text.set('\n'.join(labels))

root = tk.Tk()
App(root)
root.mainloop()
