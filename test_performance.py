#!/usr/bin/env python
# coding: utf-8
'''
This ipynotebook is modified from [qubvel\`s repo](https://github.com/qubvel/segmentation_models/tree/master/examples) 
to adjust to VOC07 dataset. As an exercise for the further implement of a custom dataset
Reqirements
- keras >= 2.2.0 or tensorflow >= 1.13
- segmenation-models==1.0.*
- albumentations==0.3.0
'''
import argparse
parser = argparse.ArgumentParser(description=
'test the prediction quality and speed on VOC-Dataset')
parser.add_argument('modelfile',type=str,
                    help='the path to the entire model file')
parser.add_argument('root_dir',type=str,
                    help='the root dir of VOC dataset, should be VOC2007 or VOC2012')
parser.add_argument('--backbone',type=str,default='efficientnetb3', 
                    help='backbone name of the model')
parser.add_argument('--output',type=str,default='output.jpg',
                    help='output file name of performance test')
args = parser.parse_known_args()[0]

import os
from os.path import join as pjoin
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt

#the root dir of the voc dataset
ROOT_DIR = args.root_dir

# class for voc dataset
# , merged some ideas from [meetshah1995](https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/pascal_voc_loader.py)

class VOCDataset:
    '''
    Dataset of VOC
    Args:
        root_dir: str, root dir of the voc dev kit
        which_split: str, one from 'train', 'val', 'test'
        classes: list, list of classes to be trained
    '''
    # PASCAL VOC 21 Classes name and their colors
    PASCAL_COLORS=[
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
   
    PASCAL_CLASSES = ['background',
                      'aeroplane',
                      'bicycle',
                      'bird',
                      'boat',
                      'bottle',
                      'bus',
                      'car',
                      'cat',
                      'chair',
                      'cow',
                      'diningtable',
                      'dog',
                      'horse',
                      'motorbike',
                      'person',
                      'pottedplant',
                      'sheep',
                      'sofa',
                      'train',
                      'tvmonitor'
                     ]
    
    def __init__(
    self,
    root_dir,
    which_split,
    classes=None,
    augmentation=None,
    preprocessing=None
    ):
        for c in classes:
            assert c in self.PASCAL_CLASSES, '\'%s\' is not a class in PASCAL07'%c
        #convert chosen class name to class index
        self.class_values = [self.PASCAL_CLASSES.index(cls.lower()) for cls in classes]
        #root dir
        self.root_dir = root_dir
        #.jpg and .png file dir
        imgs_dir = pjoin(root_dir,'JPEGImages')
        masks_dir = pjoin(root_dir,'SegmentationClass')
        #image name list .txt file dir 
        masktxt_dir = pjoin(root_dir,'ImageSets','Segmentation')
        classtxt_dir = pjoin(root_dir,'ImageSets','Main')
        #create the file id list of all images with masks
        file_path = pjoin(masktxt_dir, which_split + ".txt")
        file_list = tuple(open(file_path, "r"))
        mask_ids = [id_.rstrip() for id_ in file_list]
        
        #if only one class is chosen, create the file id list of this explict chosen class
        if len(classes)==1:
            c = classes[0]
            file_path= pjoin(classtxt_dir,c+'_'+which_split+'.txt')
            file_list = tuple(open(file_path, "r"))
            #slice op here: class string ends with '1' if exists in the image, otherwise '-1'
            class_ids = [id_[:6] for id_ in file_list if id_[7] != '-']
            self.ids = [id_ for id_ in class_ids if id_ in mask_ids]
        else:
            self.ids = mask_ids
            
        #create filepath
        self.img_filepaths = [pjoin(imgs_dir, img_id)+'.jpg' for img_id in self.ids]
        self.mask_filepaths = [pjoin(masks_dir,img_id)+'.png' for img_id in self.ids]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self,index):
        img_name = self.img_filepaths[index]
        mask_name = self.mask_filepaths[index]
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_name)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        #change class color value to coresponind class id value in the mask
        mask = self._encode_segmap(mask)
        #extracts chosen classes as one-hot code
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask
    
    def _encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.PASCAL_COLORS):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        
        return label_mask

# data augmentation
import albumentations as A

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

#define resize function for test images
def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32
    the voc dataset images have various resolution
    """
    test_transform = [
           A.Resize(320,320) 
            ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

# # Segmentation model training
import segmentation_models as sm

# segmentation_models could also use `tf.keras` if you do not have Keras installed
# or you could switch to other framework using `sm.set_framework('tf.keras')`
# not changeable once the model is trained because .h5 only contains weights
# assuming the model structure is already known
BACKBONE = args.backbone
BATCH_SIZE = 4
CLASSES = ['person']
LR = 0.0001

preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

#create model
model = keras.models.load_model(args.modelfile)

#create test dataloader
test_dataset = VOCDataset(ROOT_DIR,
                          'test',
                          classes=CLASSES,
                          augmentation=get_validation_augmentation(),
                          preprocessing=get_preprocessing(preprocess_input)
                         )

# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

import datetime
def test_performance():
    times = []
    n = 5
    fig, axn = plt.subplots(n,3,figsize=(16,5*n))
    
    ids = np.random.choice(np.arange(len(test_dataset)), size=n)

    for i, instance in enumerate(ids):
        image, gt_mask = test_dataset[instance]
        image = np.expand_dims(image, axis=0)
        start = datetime.datetime.now()
        pr_mask = model.predict(image).round()
        stop =  datetime.datetime.now()
        t = (stop-start).total_seconds()
        times.append(t)
        
        axn[i][0].imshow(denormalize(image.squeeze()))
        axn[i][1].imshow(gt_mask.squeeze())
        axn[i][2].imshow(pr_mask.squeeze())
        axn[i][0].set_title('original image')
        axn[i][1].set_title('Ground Truth Mask')
        axn[i][2].set_title('Predicted Mask')
        
    duration = np.median(times)
    FPS = 1/duration
    title = 'tested model: {} \
            average FPS: {:.2f} \
            consumed time: {:.2f}s'.format(args.modelfile,FPS,duration)
    fig.suptitle(title,fontsize='xx-large')
    fname = args.output
    fig.savefig(fname=fname)
    print('successfully saved test result as {}'.format(fname))    
    return duration,FPS

duration,FPS = test_performance()
print('average FPS:',FPS,'average consumed time:',duration)

