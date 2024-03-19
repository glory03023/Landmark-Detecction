##
Landmark Detecction base on Resnet50
used 10 landmark on custom dataset focusing fashion dataset.

##
Data preparation
dataset folder looks as following:
    dataset
    |---train
    |   |---img_x1_y1_x2_y2_...._xn_yn.jpg
    |   |---
    |
    |---test
    |

Each image should be in jpg or png format, and the file nameis seperated by '_' to show the image name and landmark point coordinates.

##
Train
    train.py

##
Teset
    eval.py
