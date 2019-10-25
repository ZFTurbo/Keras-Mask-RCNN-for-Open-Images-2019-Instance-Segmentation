## Keras Mask R-CNN for Open Images Challenge 2019: Instance Segmentation

Repository contains Mask R-CNN models which were trained on Open Images Dataset during Kaggle competition: 
https://www.kaggle.com/c/open-images-2019-instance-segmentation/leaderboard 

Repository contains the following:
* Pre-trained Mask R-CNN models (ResNet50, ResNet101 and ResNet152 backbones)
* Example code to get predictions with these models for any set of images
* Code to train your own classifier based on Keras Mask R-CNN and OID dataset 

## Requirements

Python 3.\*, Keras 2.\*, [keras-maskrcnn 0.2.2](https://github.com/fizyr/keras-maskrcnn), cv2, numpy, pandas

## Pretrained models

There are 3 Mask R-CNN models based on ResNet50, ResNet101 and ResNet152 for [300 classes](https://github.com/ZFTurbo/Keras-Mask-RCNN-for-Open-Images-2019-Instance-Segmentation/blob/master/data_segmentation/challenge-2019-classes-description-segmentable.csv). 

| Backbone | Image Size (px) | Model | Small validation mAP | LB (Public) |
| --- | --- | --- | --- | --- | 
| ResNet50 | 800 - 1024 | [521 MB](https://github.com/ZFTurbo/Keras-Mask-RCNN-for-Open-Images-2019-Instance-Segmentation/releases/download/v1.0/mask_rcnn_resnet50_oid_v1.0.h5) | 0.5745 | 0.4259 |
| ResNet101 | 800 - 1024 | [739 MB](https://github.com/ZFTurbo/Keras-Mask-RCNN-for-Open-Images-2019-Instance-Segmentation/releases/download/v1.0/mask_rcnn_resnet101_oid_v1.0.h5) | 0.5917 | 0.4345 |
| ResNet152 |  800 - 1024 | [918 MB](https://github.com/ZFTurbo/Keras-Mask-RCNN-for-Open-Images-2019-Instance-Segmentation/releases/download/v1.0/mask_rcnn_resnet152_oid_v1.0.h5) | 0.5899 | 0.4404 |

* Model - can be used to resume training or can be used as pretrain for your own instance segmentation model

## Inference 

Simple example can be found here: [inference_example.py](https://github.com/ZFTurbo/Keras-Mask-RCNN-for-Open-Images-2019-Instance-Segmentation/blob/master/inference_example.py)

## Training

For training you need to download OID dataset (~500 GB images): https://storage.googleapis.com/openimages/web/download.html
You need all images, all masks and all CSV-files related to Instance Segmentation track. 

Then run script (change parameters and file locations at the bottom of script):
* [training/train_maskrcnn.py]()
