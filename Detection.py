
from keras.models import load_model
import cv2
import numpy as np
from pygame import mixer
import pyfirmata.util
import keyboard
import time
import math
from gpiozero import Buzzer
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import *
from tkinter import ttk

mixer.init()
sound = mixer.Sound('alarm.wav')
model = load_model('BestSelectedModel.model')
labels_dict = {0: 'No MASK', 1: 'MASK'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
cap = cv2.VideoCapture(0)
board = pyfirmata.Arduino('COM3')
print("Communication Successfully started")
SENSOR_PIN = 0
buzz_pin = board.get_pin('d:9:o')

it = pyfirmata.util.Iterator(board)
it.start()
board.analog[SENSOR_PIN].enable_reporting()
buzzer = Buzzer(9)
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for x in range(1):
         resized = cv2.resize(gray, (224, 224))
         normalized = resized / 255.0
         reshaped = np.reshape(normalized, (1, 224, 224, 1))

         result = model.predict(reshaped)

         label = np.argmax(result, axis=1)[0]

         #cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 4)
         #cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], 4)
         cv2.putText(img, labels_dict[label], (50, 30), cv2.FONT_ITALIC, 1, (255, 0, 0), 4)

         if (labels_dict[label] == 'MASK'):
             board.digital[13].write(0)
             board.digital[12].write(1)
             buzz_pin.write(0)
             time.sleep(1)
             print("Beep")
         elif (labels_dict[label] == 'No MASK'):
             board.digital[12].write(0)
             board.digital[13].write(1)
             buzz_pin.write(1)
             sound.play()
             time.sleep(1)
             print("NO Beep")

    if (keyboard.is_pressed('p')):
        board.digital[13].write(0)
        board.digital[12].write(1)
        buzz_pin.write(0)
        time.sleep(10)
        print('pass')



    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()