#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import subprocess
import picamera
import json
import chainer
import argparse
import numpy as np
import math
from chainer import cuda
import chainer.functions as F
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
from chainer import serializers
import os

sys.path.append('./code')
from CaptionGenerator import CaptionGenerator

camera = picamera.PiCamera()
camera.resolution = (224, 224)

devnull = open('os.devnull', 'w')

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu",default=-1, type=int, help=u"GPU ID.CPU is -1")
parser.add_argument('--beam',default=3, type=int,help='beam size in beam search')
parser.add_argument('--depth',default=50, type=int,help='depth limit in beam search')
parser.add_argument('--lang',default="<sos>", type=str,help='special word to indicate the langauge or just <sos>')
args = parser.parse_args()

caption_generator=CaptionGenerator(
    rnn_model_place='./data/caption_en_model40.model',
    cnn_model_place='./data/ResNet50.model',
    dictonary_place='./data/MSCOCO/mscoco_caption_train2014_processed_dic.json',
    beamsize=args.beam,
    depth_limit=args.depth,
    gpu_id=args.gpu,
    first_word= args.lang,
    )

i = 1
while True:
    camera.capture('output/capture.jpg')
    captions = caption_generator.generate('output/capture.jpg')
    word = "Okay I got it. I think I'm seeing " + " ".join(captions[0]["sentence"][1:-1])
    ws = "_".join(captions[0]["sentence"][1:-1])
    os.rename('output/capture.jpg', 'output/{}-{}.jpg'.format(i,ws))
    print(word)
    subprocess.run(['pico2wave', '-w', 'output/{}-result.wav'.format(i), "{}".format(word)], 
                   stdout=devnull, stderr=subprocess.STDOUT)
    subprocess.call(['aplay','output/{}-result.wav'.format(i)])
    i += 1
    # !pico2wave -w lookdave.wav "Look Dave, I can see you're really upset about this." && aplay lookdave.wav