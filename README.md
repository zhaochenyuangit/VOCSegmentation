## How to run unet with tensorflow

This markdown file is a manual about how to modify the data-loader of [this great Github repo](https://github.com/qubvel/segmentation_models) to use segmentation model with custom datasets. And then deploy the model on jetson nano.

<img src=".\img\notsquare2-1590757864650.png" alt="preview" style="zoom:72%;" />

### Workflow Overview

[TOC]

### Introduction to 2 common public dataset

##### 1. CamVid Dataset

CamVid is a car camera live-stream Dataset for semantic segmentation from Cambridge. The [original CamVid Dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) has 32 classes, and the mask is painted with color. 

In the [example ipy-notebook](https://github.com/qubvel/segmentation_models/blob/master/examples/binary%20segmentation%20(camvid).ipynb), however, the author used [a modified version](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid) of the Dataset. In the modified version, class numbers are reduced to 12, and the data mask is no longer painted with color, but each pixel in the mask directly has the value of its class number from 0 to 11. Therefore the mask looks almost black, no longer human-readable but easy for program to extract masks in one-hot code form of a certain class.  

<img src=".\img\CamVid.png" alt="camvid"  />

Following is a fraction of the modified CamVid mask, by replacing class specified color with class number, the details becomes hard to see, but we can easily extract the pixel location of a desired class by:

```python
# pseudo code
mask = (mask == v) # for one class
masks = [(mask == v) for v in class_values] # for multiple classs
```

<img src=".\img\CamVidMask.PNG" style="zoom: 25%;" />

which returns a one-hot code mask of the desired class.

File structure of the modified CamVid Dataset is shown as below:

```python
/CamVid
|---/train
|---/trainannot
|---/val
|---/valannot
|---/test
|---/testannot
```

Here is the summary of CamVid Dataset.

| item        |                                                              |
| :---------- | ------------------------------------------------------------ |
| image size  | $480\times360$ pixels                                        |
| data volumn | 367 train，101 val，233 test                                 |
| mask style  | value of each pixel in the mask is its corresponding class number |
| Classes     | ['sky', 'building', 'pole', 'road', 'pavement','tree', 'sign symbol', 'fence', 'car','pedestrian', 'bicyclist', 'unlabeled']<br>12 classes in total |

it is a small dataset because it only has several hundreds images. But the fancy part is that CamVid Dataset has a fixed image size, and almost all classes shows up in every single image.  Therefore, no matter we want to train binary- or multiple segmentation classes, almost all images are useful and no need to exclude some images. 

##### 2. VOC Datasets

**V**isual **O**bject **C**hallenge Dataset is used for the competition of PASCAL held from year 2005 to 2012. The Dataset has only 4 classes in 2005, so the dataset is not comprehensive nor useful until 2007. In the next year, PASCAL create another dataset VOC 08. And since then, new dataset is created by adding new images to the previous one, until 2012, when the last competition is held.

![PASCAL VOC](https://arleyzhang.github.io/articles/1dc20586/1523028261517.png)



Therefore, VOC 07 and VOC 12 are two totally different dataset. We could combine these two to have a larger dataset. Or train on the whole VOC 12 and use VOC 07 as test set. 

Following is the hierarchy of VOC 07 Dataset: 

```python
/VOCdevkit
|---/VOC2007
	|---/Annotations # xml file for object detection
	|---/ImageSets	# image ID lists of certain group as txt file
		|---/Layout # for object detection
		|---/Main	# id lists of a certain class
		|---/Segmentation # id lists of all images having a mask
        	|---/test.txt # split into 3 subsets
            |---/train.txt
            |---/val.txt
    |---/JPEGImages	#original images
	|---/SegmentationClass # masks for semantic segmentation
	|---/SegmentationObject	# for instance segmentation
```

> Please note that the test data of VOC 12 is **NOT** released.

| item       |                                    |
| ---------- | ---------------------------------- |
| image size | various                            |
| mask style | painted with class-specified color |
| Classes    | 21 classes                         |

Since the mask is painted with color, we must first convert the color to its corresponding class number, like the above CamVid Dataset. 

<img src=".\img\VOC.png" alt="VOC"  />

following is the class names and their corresponding colors in RGB order:

```
background : [0, 0, 0]
aeroplane : [128, 0, 0]
bicycle : [0, 128, 0]
bird : [128, 128, 0]
boat : [0, 0, 128]
bottle : [128, 0, 128]
bus : [0, 128, 128]
car : [128, 128, 128]
cat : [64, 0, 0]
chair : [192, 0, 0]
cow : [64, 128, 0]
diningtable : [192, 128, 0]
dog : [64, 0, 128]
horse : [192, 0, 128]
motorbike : [64, 128, 128]
person : [192, 128, 128]
pottedplant : [0, 64, 0]
sheep : [128, 64, 0]
sofa : [0, 192, 0]
train : [128, 192, 0]
tvmonitor : [0, 64, 128]
```

We could extract the desired class by:

```python
# pseudo code
import numpy as np
omask = cv2.imread('orignal_mask.png')
omask = cv2.cvtColor(omask, cv2.COLOR_BGR2RGB)
def encoder(omask):
    '''
        omask: (H,W,3) array, image
        color: (1,3) array, RGB color
        
        return: label mask, a (H,W) array
        '''
    label_mask = np.zeros((mask.shape[0], mask.shape[1]))
    for i, color in enumerate(PASCAL_COLORs):
        # (H,W) array, True if pixel color equals class color
        filterMask = np.all(omask == color, axis=-1)
        # tuple of 2 array, the x and y location of the True Value on the filterMask
        locations = np.where(filterMask)
        # fill the location in label mask with class number
        label_mask[locations] = i
        '''
        array([  0,   0,   0, # unrelated location remains unchanged
              1,   1,   1, 	  # related location changed to corresponding number
              1,   9,   9,]	  
        '''
        
    return label_mask
```

Another problem that occurs in VOC Dataset is that not every class appears in every image. For example, the image above only have 3 classes (including background) while we have 21 classes in total. This may affect our training result negatively in binary segmentation task. If we want our model to recognize aeroplane in the image above, the mask for aeroplane will be totally black because there is no plane. We don't want to generate such useless masks and pollute our training data.

The solution is to only load the images that contains the classes we concern.

<img src=".\img\VOCoverview.png" alt="VOC overview"  />

Above is an overview of all trainable images with mask in a semantic segmentation task. The bar plot order for each class is train, val and test (VOC 2012 no test dataset). For binary segmentation, we need to choose one single class for training. The `human` class is therefore chosen, because it has much more images than others.

### Annotation of custom images

[labelme](https://github.com/wkentaro/labelme) is a great tool to create VOC-style annotations for semantic segmentation.

##### installation

Ubuntu 

```shell
sudo apt-get install labelme
```

Anaconda

```shell
# python3
conda create --name=labelme python=3.6
conda activate labelme
pip install labelme
```

##### open up the GUI

```shell
# in shell
labelme
```

<img src=".\img\labelmeGUI2.PNG" style="zoom:43%;" />

A blank GUI will then pop out. Click "Open Dir" to choose the original image file-path. Then click the "Create Polygons" to circumscribe the objects.

> When creating the polygons, we can zoom-in and zoom-out by holding Ctrl+Scroll wheel. 

After creating the polygon we need to assign a class to the mask. Click save will create a .json file with the same filename as image in the same folder, the .json file play a role as "meta" annotation. Later we can generate PNG mask from the json file.

###### Alternative

```shell
# in shell
labelme image_foler --labels labels.txt
```

will open a GUI with a list of available classes defined in the txt file and open the image folder at the same time.

##### export VOC-format dataset for semantic segmentation

```shell
# in shell
python ./labelme2voc.py imagefolder destinationfoldervoc --labels labels.txt
#generates
#   orginal images	- destinationfoldervoc/JPEGImages						
#   .npy files		- destinationfoldervoc/SegmentationClass
#   PNG masks 		- destinationfoldervoc/SegmentationClassPNG
#   human reading masks	- destinationfoldervoc/SegmentationClassVisualization
```

The .npy files and .png files correspond to the two mask styles mentioned above.

* The label as .npy files (numpy array) contains only very low label values for class number (ex. `0, 4, 14`), and `255` indicates the `__ignore__` label value (`-1` in the npy file).
* The label as .png files are painted with color, so easy for human to read. 

##### Mask color

When exporting the masks in voc-format, it is not possible to change the color map. In other words, the class number and the mask color is bounded. The mask color is always generated in the following order:

<img src="C:\Users\zhaoc\Documents\MyUnet\img\colormap.png" alt="colormap" style="zoom:50%;" />

In `label.txt` we should always have `__ignore__` as the first (-1) class, `_background_` as the second class (0). The colormap can be inspected by:

```python
import imgviz
imgviz.imgviz.label_colormap(n_label=21, value=None)
```

But if we still want to modify custom colormap, we should look into `labelme/labelme/utils/_io.py`, in function `lblsave` (label save), we can change the colormap.

### Write a custom dataset class

A dataset is usually a subset of the original images, split into train-, validation- and test-set. The dataset class stores the file-path to all its images.  

Usually, the image folder path and image ids are stored separately. For example, `./test/images` is the path to image folder, and `./test/masks` is the path to mask folder. While the id-list is a python list contains all image file names in that folder, for example `[001.png, 002.png, ...]`. By combination, we get `./test/images/001.png`. It is important to keep the image-ID for a certain image and its mask to be the same.

> if the images are already split into train/val/test set like CamVid, we cam obtain the list of file-IDs by `file_list = os.listdir(images_dir)`, which returns all filenames in the folder as a list.
>
> if the images are not split, but there is a txt file including the split infomation like VOC, we can obtain the list by `file_list = tuple(open(txt_file_path, "r"))`

The custom dataset should include the following methods:

- `__len__` so that `len(dataset)` returns the size of the dataset.
- `__getitem__` to support the indexing such that `dataset[i]` can be used to get i_th sample

the dataset class doesn't store the images in memory, but only stores the relative path to all its images. The images will not be loaded when the dataset class is created, but is called with `__getitem__` method, e.g.  `image, mask = dataset[0]`.

The returned `image` variable of the `__getitem__` method is the original jpg image with the size of `(H,W,3)`

The returned `mask` variable is a series of one-hot code mask of size `(H,W)` stacked on a new axis, with size of `(H,W,N)`, where `N` is the number of extracted classes. 

```python
# pseudo code
def __getitem__:	
    img = cv2.imread('image.jpg')
    mask = cv2.imread('mask.png')
    mask = encoder(mask)
    masks = [(mask == v) for v in class_values]
    mask = np.stack(masks, axis=-1).astype('float')
    
    return img, mask
```

To get the mask of a certain class:

```python
# get the mask of the first desired class
first_mask = mask[...,0] # select by the last axis
```

##### Dataloader

After writing the dataset class we can then pass it to the dataloader. The dataloader stacks several image-mask pair to create a batch.

```python
# pseudo code
BATCH_SIZE = 4
def __getitem__:
    data = [(img1,msk1), (img2,msk2), (img3,msk3), (img4,msk4)]
    batch = [np.stack(samples, axis=0) for samples in zip(*data)]
    
    return batch
'''
batch[0].shape = (BATCH_SIZE,H,W,3) is the batch of images
batch[1].shape = (BATCH_SIZE,H,W,N) is the batch of masks
'''
```

 the length of dataloader must be multiple times of BATCH_SIZE. We discard the last image-mask pairs by `//` operation, if the length cannot be divided.

```python
#pseudo code
def __len__(self):
    
    return len(self.indexes) // self.batch_size
```

### Train model on PC

```
import segmentation_models as sm
```

+ set the constants to avoid typo in the following program

```python
BACKBONE = 'efficientnetb3'
BATCH_SIZE = 8
CLASSES = ['car']
N_CLASSES = len(CLASSES)
ACTIVATION = 'sigmoid' if N_CLASSES == 1 else 'softmax'
LR = 0.0001 # Learning rate
EPOCHS = 40
```

+ create model

```python
model = segmentation_models.Unet(
    backbone_name=BACKBONE, 
    input_shape=(None, None, 3),#image of any size, 3 channels
    classes=N_CLASSES, 			#output classes number
    activation='sigmoid', 		#set to 'softmax' if multiple classes	
    weights=None, 				#load the .h5 weights if re-training
    encoder_weights='imagenet', #pretrained on on 2012 ILSVRC ImageNet dataset
    encoder_freeze=True, 		#set to Ture if Transfer Learning
)
```

* compile the model

'compile' means define the optimizer and loss function for the model. A model cannot be trained without compilation.

```python
# optimizer and loss are indispensible
# metrics is not necessary
model.compile(optimizer = keras.optimizers.Adam(LR),
              loss = sm.losses.DiceLoss(),
              metrics =[sm.metrics.IOUScore(threshold=0.5),
                        sm.metrics.FScore(threshold=0.5)]
             )
```

* define callback functions

callbacks are functions that will be called at the start or end of specific training stages. The method name of a callback function must be the keywords predefined by `keras`.

One can define a custom callback function by:

```python
class MyCustomCallback(tf.keras.callbacks.Callback):
	def on_(train|test|predict)_(begin|end)(self, logs=None):
		do_something() #at the begin or end of fit/evaluate/predict
        
    def on_train_batch_end(self, batch, logs=None):
        #batch is the current batch number
        do_something() #at the end of training this batch
```

see the [keras callback document](https://www.tensorflow.org/guide/keras/custom_callback) for more details.

In our task, we can use the ModelCheckpoint callback class to save the model at the end of each epoch.

```python
callbacks = [
    keras.callbacks.ModelCheckpoint('./best_model_mobilnetv2.h5',
                                    save_weights_only=True,
                                    save_best_only=True,
                                    mode='min'),
]
# include epoch info in the filename
weights_filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
callbacks = [
    keras.callbacks.ModelCheckpoint(weights_filepath,
                                    save_weights_only=True,
                                    save_best_only=True,
                                    mode='min'),
]
# save the whole model instead of weights-only
# in callback function it is realized by replacing model.save_weights() with
# model.save()
callbacks = [
    keras.callbacks.ModelCheckpoint(weights_filepath,
                                    save_weights_only=False,
                                    save_best_only=True,
                                    mode='min'), 
]
'''
When the parameter 'save_best_only' is set to True. The model will not be saved unless it has a better score than before.
If a lost function is passed to model as creteria, parameter 'mode' should be set to 'min', since a lower loss is better;
If accuracy is passed to model, 'mode' should be 'max', because a higher accuracy is better.
'''
```

* train model

```python
history = model.fit_generator(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=10,#EPOCHS
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
)
```

Beside the history returned by fit_generator(), a tensorboard callback function is also great for visualization of the training process.

* save model

the weights-only file is already saved by the callback function during training. 

To save the entire model, we can either save it at the end of training stage by :

```python
# cover the last weights with the best weights
model.load_weights('./best/weights.h5')
# now save the model
model.save('./path/entire/model.h5')
```

Or save its architecture only as json file:

```python
import json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
```

 And then combine the json file and h5 file to create an entire model:

```python
# later...
# load json and create model
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    
loaded_model = model_from_json(loaded_model_json)
# load best weights and save entire model
loaded_model.load_weights('./best/weights.h5')
model.save('./path/entire/model.h5')
```

### Deploy model on jetson

##### load the model from weights-only file

When restoring a model from weights-only, we must have a model with the same architecture as the original model. 

```python
# create a same same model with exact same settings
model = segmentation_models.Unet(
    backbone_name=BACKBONE, 
    input_shape=(None, None, 3),#image of any size, 3 channels
    classes=N_CLASSES, 			#output classes number
    activation='sigmoid', 		#set to 'softmax' if multiple classes	
    weights=None, 				#load the .h5 weights if re-training
    encoder_weights='imagenet', #pretrained on on 2012 ILSVRC ImageNet dataset
    encoder_freeze=True, 		#set to Ture if Transfer Learning
)
# and compile the model again if want retrain on jetson
# but the compile configuration don't need to be the same
# compilation is not necessary if only used for predict
model.compile(optimizer = keras.optimizers.Adam(LR),
              loss = sm.losses.DiceLoss(),
              metrics =[sm.metrics.IOUScore(threshold=0.5),
                        sm.metrics.FScore(threshold=0.5)]
             )
```

The pre-trained weights can be loaded by 'weights' parameter when create the model. Or by `model.load_weights('./path/to/.h5')` after create the model.

##### load the model from entire model file

When restore the model from an entire model file saved with `model.save('./path/to/.h5')`. It is no longer needed to define the model again. However, in the original repo there are two custom layers not defined in keras: swish, FixedDropout.

To define these two layers one can either import segmentation_models packages:

```python
import segmentation_models as sm
new_model = tf.keras.models.load_model('my_model.h5')
```

Or define these layers by oneself:

```python
import tensorflow as tf
from keras import backend as K
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
    		return self.noise_shape
    symbolic_shape = K.shape(inputs)
    noise_shape = [symbolic_shape[axis] if shape is None else shape for axis, shape in enumerate(self.noise_shape)]
    return tuple(noise_shape)

model = tf.keras.models.load_model(path,compile=False,custom_objects{
'swish':tf.compat.v2.nn.swish,
'FixedDropout':FixedDropout})
```

##### image preprocessing

Pre-processing is indispensable if want to obtain a decent predict result. 

Firstly, we need to resize the image size to multiple times of 32. Because in Unet architecture, there are 5 2X Pooling layers, the original image will be resized to 1/32. We could use `Albumentation.PadIfNeeded` function to pad the image.

```python
# pseudo code
import Albumentation as A
from math import ceil
H,W,C = image.shape
H = 32*ceil(H/32)
W = 32*ceil(W/32)
aug = A.Compose([A.PadIfNeeded(H,W)])
sample = aug(image=image)
image = sample['image']
```

After padding the image, we need to process the image with the preprocessing function offered by segmentation_models package.

```python
# pseudo code
import segmentation_models as sm
BACKBONE = 'efficientnetb3'
preprocess_input = sm.get_preprocessing(BACKBONE)
pre = A.Compose([
        A.Lambda(image=preprocess_input)
        ])
sample = pre(image=image)
image = sample['image']
```

##### predict

Prediction is done by one line `model.predict`. Similar as the training stage, the size of input image is `(BATCHSIZE,H,W,C)`, the size of output masks is `(BATCHSIZE,H,W,N)`, where `BATCHSIZE` should always be 1 in prediction case, and `N` is the number of classes to predict, in binary segmentation case should be 1. 

```python
# add the BATCHSIZE axis to the (H,W,3) image
# so that the input image becomes (1,H,W,3) in size
image = np.expand_dims(image, axis=0)
pr_mask = model.predict(image).round()
pr_mask = pr_mask[...,0] # select the first class on the mask axis
pr_mask = pr_mask.squeeze() # squeeze out the batch axis to get (H,W) scala mask
```



