import os
import tkinter as tk
from tkinter.filedialog import askopenfilename

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras import models

from utils import decoder

IMAGE_FORMATS = [("JPEG", "*.jpg"), ("PNG", "*.png")]
SIZE = 512
ZOOM_INC = 2**(1/2)
MAX_ZOOM = 2**(3/2)
SLIDE_STEP = 8
FINAL_SHAPE = 128, 64, 1
MODEL_PATH = 'models/anpr.h5'

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

      self.load_button = tk.Button(master, text="Load image", command=self.predict)
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
    w, h = self.image.size
    w, h = (SIZE, h * SIZE // w) if w > h else (w * SIZE // h, SIZE)
    self.display_image = ImageTk.PhotoImage(self.image.resize((w, h), Image.ANTIALIAS))
    self.image_label.configure(image=self.display_image)
    return np.array(self.image.convert('L'), np.uint8), w, h

  def predict(self):
    image, display_w, display_h = self.load_image()
    h, w = image.shape
    zoom = 1
    cropped, bboxes, valid = [], [], []
  
    while zoom <= MAX_ZOOM:
      width, height = int(w * zoom), int(h * zoom)
      overflow_x, overflow_y = width - w, height - h
      coeff = w / width

      scaled = cv.resize(image, (width, height))

      for i in range(0, int(overflow_x) + SLIDE_STEP, SLIDE_STEP):
        for j in range(0, int(overflow_y) + SLIDE_STEP, SLIDE_STEP):
          bboxes.append(((int(i*coeff), int(j*coeff)), (int(i*coeff + w*coeff), int(j*coeff + h*coeff))))
          cropped.append(cv.resize(scaled[j:j+h, i:i+w], FINAL_SHAPE[:-1]).reshape(FINAL_SHAPE))

      zoom *= ZOOM_INC

    predictions = self.model.predict(np.array(cropped))

    img = np.array(self.image)
    for i, prediction in enumerate(predictions):
      code = decoder(prediction)
      if code[:1] == '1':
        valid.append(code[3:])
        img = cv.rectangle(img, *bboxes[i], (255, 255, 255))

    self.display_image = ImageTk.PhotoImage(Image.fromarray(img).resize((display_w, display_h), Image.ANTIALIAS))
    self.image_label.configure(image=self.display_image)
    self.prediction_text.set('\n'.join(valid))

root = tk.Tk()
App(root)
root.mainloop()
