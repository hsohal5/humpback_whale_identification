# Humpback whale Identificcation

## Abstract

This report describes about the current work done on whale photo surveillance system. 
We are developing a deep learning model using convolutional neural network to be used to identify the type of humpback whale. 
The whale species are identified by the light and dark pigmentation patches on their tails.
The data set it taken from Kaggle competition.

## File Description:
PHash.py : In this file, we remove the duplicate images using phash algorithm.
imageaugumentation.py : This file is for creating more images of classes which has less than 15sampels per class.
cropimage.py : This is for cropping the images.. there is an extra water portion,so that extra water portion has to cut and only the fluke is present.
model.py : We build the model using CNN and ResNets.

## Libraries:
numpy;
pandas;
opencv;
sklearn;
scipy;
matplotlib;
keras;
Tensorflow;
ResNets;
