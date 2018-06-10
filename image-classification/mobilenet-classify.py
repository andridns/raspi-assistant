import sys
import os
import subprocess
import picamera
import numpy as np
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions

camera = picamera.PiCamera()
camera.resolution = (224, 224)

model = MobileNet(weights='imagenet')
devnull = open('os.devnull', 'w')

i = 1
while True:
    camera.capture("img.jpg")
    img = image.load_img("img.jpg", target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    pred = model.predict(x)
    word = decode_predictions(pred)[0][0][1].replace('_',' ')
    word2 = decode_predictions(pred)[0][1][1].replace('_',' ')
    speak = "Okay I got it. I believe I'm looking at some kind of {} or {}".format(word, word2)
    print(speak)
    os.rename('img.jpg', 'outputs/mobilenet/{}-{}-{}.jpg'.format(i,word, word2))
    subprocess.run(['pico2wave', '-w', 'outputs/mobilenet/{}-result.wav'.format(i), "{}".format(speak)], 
                   stdout=devnull, stderr=subprocess.STDOUT)
    subprocess.call(['aplay','outputs/mobilenet/{}-result.wav'.format(i)])
    print()
    i += 1
