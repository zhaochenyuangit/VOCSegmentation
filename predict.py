#!/usr/bin/env python
# coding: utf-8
'''
predict a single image using the full model file
from [qubvel\`s repo](https://github.com/qubvel/segmentation_models/tree/master/examples)
Reqirements
- keras >= 2.2.0 or tensorflow >= 1.13
- segmenation-models==1.0.*
- albumentations==0.3.0
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if os.path.exists('predict'):
    pass
else:
    os.mkdir('predict')
from os.path import join as pjoin
import argparse
parser = argparse.ArgumentParser(description=
                                 'test a single image')
parser.add_argument('modelfile',type=str,
                    help='the path to the entire model file')
parser.add_argument('--backbone',type=str,default='efficientnetb3',
                    help='model archtecture (default: efficientnetb3)')
parser.add_argument('input_image',type=str,
                    help='the path to the input image')
parser.add_argument('--output',type=str,default='predict/output',
                    help='path to the output directory')
args = parser.parse_args()

    
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import segmentation_models as sm

# rgba format, add alpha channel to transparent the background
MASKCOLOR=[128,0,0,0.5]

def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

#create model
model = keras.models.load_model(args.modelfile)

preprocess_input = sm.get_preprocessing(args.backbone)
#predict an arbitary image
image = cv2.imread(args.input_image)
# pre-processing, simplified from the original ipynb file
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
aug = A.Compose([
           A.Resize(320,320) 
            ])
sample = aug(image=image)
image = sample['image']
pre = A.Compose([
        A.Lambda(image=preprocess_input)
        ])
sample = pre(image=image)
image = sample['image']
# show the image and save it
plt.figure()
plt.imshow(denormalize(image.squeeze()))
image = np.expand_dims(image, axis=0)
pr_mask = model.predict(image).round()
pr_mask = pr_mask[...,0] # select the first class on the mask axis
pr_mask = pr_mask.squeeze() # squeeze out the batch axis
# calculate mask area
H, W = pr_mask.shape
mask_percentage = (pr_mask.sum()/(H*W))*100
plt.text(0,1,'Predicted Image:{}\n Mask Area:{:.2f}%'.format(args.input_image,mask_percentage))
pr_mask = pr_mask[...,np.newaxis] # add a newaxis for convert to rgba format
pr_mask=np.dot(pr_mask,np.array(MASKCOLOR)[np.newaxis,...])
plt.imshow(pr_mask,interpolation=None)
plt.savefig(args.output)
print('save file successfully')
