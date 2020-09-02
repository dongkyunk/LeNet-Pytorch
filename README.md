# Pytorch implementation of [**LeNet**](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
This is the first paper I am review/implementing for my project: "Ten Papers && Ten Implementations"


## Overview 
LeNet-5 consists of two sets of convolutional and average pooling layers, followed by a flattening convolutional layer, then two fully-connected layers and finally a Gaussian classifier.


<img src="https://blog.kakaocdn.net/dn/bwmQbA/btqB8PCH3IE/lCT2DAeSNV2rbVlIGBUCh0/img.png" width = "100%">

## How To Use
Clone this repo
```
git clone https://github.com/dongkyuk/LeNet
```
Go to the repo directory
```
cd LeNet/
```
Install requirements
```
pip install -r requirements.txt
```
Start Training 
```
python train.py --optim adam --epoch 50
```
You can change the optimizer, epoch size, etc using the parse options. Take a look at opt.py in the utils folder.
Please note that it doesn't support loss function choosing options yet.

## Netron Model Image

![Model](https://github.com/dongkyuk/LeNet-Pytorch/blob/master/lenet.png)

## Reference

https://engmrk.com/lenet-5-a-classic-cnn-architecture/

## To Do
Add Tensorboard feature
