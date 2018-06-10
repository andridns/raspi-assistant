# image caption generation by chainer 

This repository contains an implementation of typical image caption generation based on neural network (i.e. CNN + RNN). The model first extracts the image feature by CNN and then generates captions by RNN. CNN is ResNet50 and RNN is a standard LSTM .

## requirements
chainer 1.19.0  http://chainer.org 
```
#After installing anaconda, you can install chainer, sepcifically, version1.19.0,  in this way. 
pip install chainer==1.19.0 
```
