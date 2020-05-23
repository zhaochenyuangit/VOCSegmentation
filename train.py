#!/usr/bin/env python
# coding: utf-8

# ### Reqirements
# - keras >= 2.2.0 or tensorflow >= 1.13
# - segmenation-models==1.0.*
# - albumentations==0.3.0

# # Loading dataset

# This ipynotebook is modified from [qubvel\`s repo](https://github.com/qubvel/segmentation_models/tree/master/examples) to adjust to VOC07 dataset. As an exercise for the further implement of a custom dataset


import os
from os.path import join as pjoin
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = pjoin('.','data','VOCdevkit','VOC2007')


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


# class for voc2007 dataset
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
            class_ids = [id_[:6] for id_ in file_list if id_[7] is not '-']
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

#dataloader class
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

# data augmentation
import albumentations as A

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.RandomCrop(height=320, width=320, always_apply=True),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)

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
BACKBONE = 'efficientnetb3'
BATCH_SIZE = 4
CLASSES = ['person']
LR = 0.0001
EPOCHS = 40

preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

#create model
#TODO: load pretrained weights if checkpoint exists, otherwise start from new
if 
    model = sm.Unet(BACKBONE,
                    classes=n_classes,
                    activation=activation,
                   encoder_weights='imagenet',
                   encoder_freeze=True)
else:

#define optomizer
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)

# Dataset for train images
train_dataset = VOCDataset(ROOT_DIR,
                        'train',
                        classes=CLASSES,
                        augmentation=get_training_augmentation(),
                        preprocessing=get_preprocessing(preprocess_input))

# Dataset for validation images
valid_dataset = VOCDataset(ROOT_DIR,
                        'val',
                        classes=CLASSES,
                        augmentation=get_training_augmentation(),
                        preprocessing=get_preprocessing(preprocess_input))

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

# check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, 320, 320, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, 320, 320, n_classes)

# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint('./best_model.h5',
                                    save_weights_only=True,
                                    save_best_only=True,
                                    mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
]

# train model
history = model.fit_generator(
                train_dataloader, 
                steps_per_epoch=len(train_dataloader), 
                epochs=EPOCHS, 
                callbacks=callbacks, 
                validation_data=valid_dataloader, 
                validation_steps=len(valid_dataloader),
)

# Plot training & validation iou_score values
plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#create test dataloader
test_dataset = VOCDataset(ROOT_DIR,
                          'test',
                          classes=CLASSES,
                          augmentation=get_validation_augmentation(),
                          preprocessing=get_preprocessing(preprocess_input)
                         )

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

# load best weights
model.load_weights('best_model.h5') 

scores = model.evaluate_generator(test_dataloader)

print("Loss: {:.5}".format(scores[0]))
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))

n = 5
ids = np.random.choice(np.arange(len(test_dataset)), size=n)

for i in ids:
    
    image, gt_mask = test_dataset[i]
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image).round()
    
    visualize(
        image=denormalize(image.squeeze()),
        gt_mask=gt_mask[..., 0].squeeze(),
        pr_mask=pr_mask[..., 0].squeeze(),
    )

#predict an arbitary image
#image = cv2.imread('./data/ZHAO_20190401_211235.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
aug = get_validation_augmentation()
sample = aug(image=image)
image = sample['image']
pre = get_preprocessing(preprocess_input)
sample = pre(image=image)
image = sample['image']

image = np.expand_dims(image, axis=0)
pr_mask = model.predict(image).round()
visualize(
        image=denormalize(image.squeeze()),
        pr_mask=pr_mask[..., 0].squeeze(),
    )

