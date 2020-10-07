import os
import tkinter as tk
from tkinter.filedialog import askopenfilename

from PIL import Image, ImageTk

IMAGE_FORMATS = [("JPEG", "*.jpg")]
SIZE = 512

class App:
  def __init__(self, master):
    self.master = master
    master.title("Automatic Number-Plate Recogintion")

    self.image_text = tk.StringVar()
    self.image_name = tk.Label(master, textvariable=self.image_text)
    self.image_name.pack()

    self.image_label = tk.Label(master)
    self.image_label.pack()

    self.prediction_text = tk.StringVar()
    self.prediction_label = tk.Label(master, textvariable=self.prediction_text)
    self.prediction_label.pack()

    self.load_button = tk.Button(master, text="Load image", command=self.load_and_predict)
    self.load_button.pack()

  def load_and_predict(self):
    # Load
    path = askopenfilename(filetypes=IMAGE_FORMATS)
    self.image_text.set(os.path.split(path)[1])
    self.image = Image.open(path)
    w, h = self.image.size
    w, h = (SIZE, h * SIZE // w) if w > h else (w * SIZE // h, SIZE)
    self.image = self.image.resize((w, h), Image.ANTIALIAS)
    self.image_label.configure(image=ImageTk.PhotoImage(self.image))
    # Predict
    self.prediction_text.set('')

root = tk.Tk()
App(root)
root.mainloop()
