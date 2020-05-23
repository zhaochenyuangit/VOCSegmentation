## How to run unet with tensorflow

This markdown file is a manual about how to modify the data-loader of [this great Github repo](https://github.com/qubvel/segmentation_models) to use segmentation model with custom datasets. And then deploy the model on jetson nano.

#### workflow overview

+ data annotation of custom training images
+ build the custom dataset class

+ train the model on computer
+ run the model on jetson nano for performance test and segmentation

#### the structure of 2 common public dataset

##### 1. CamVid Dataset

CamVid is a car camera live-stream Dataset for semantic segmentation from Cambridge. The [original CamVid Dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) has 32 classes, and the mask is painted with color. 

In the example ipy-notebook, however, the author used [a modified version](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid) of the Dataset. In the modified version, class numbers are reduced to 12, and the data mask is no longer painted with color, but each pixel in the mask directly has the value of its class number from 0 to 11. Therefore the mask looks almost black, no longer human-readable but easy for program to extract masks in one-hot code form of a certain class.  

![](C:\Users\zhaoc\Documents\git\segmentation_models\img\CamVid.png)

Following is a fraction of the modified CamVid mask, by replacing class specified color with class number, the details becomes hard to see, but we can easily extract the pixel location of a desired class by:

```python
# pseudo code
mask = (mask == v) # for one class
masks = [(mask == v) for v in class_values] # for multiple classs
```

<img src="C:\Users\zhaoc\Documents\git\segmentation_models\img\CamVidMask.PNG" style="zoom: 25%;" />

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
		|---/Layout 	# for object detection
		|---/Main	# id lists of a certain class
		|---/Segmentation # id lists of all images having a mask
        	|---/test.txt	# split into 3 subsets
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

![VOC](C:\Users\zhaoc\Documents\git\segmentation_models\img\VOC.png)

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
omask = cv2.imread('orignal mask')
for i, color in enumerate(PASCAL_COLORs):
    
    
```



Another problem that occurs in VOC Dataset is that not every class appears in every image. For example, the image above only have 3 classes (including background) while we have 21 classes in total. This may affect our training result negatively in binary segmentation task. If we want our model to recognize aeroplane in the image above, the mask for aeroplane will be totally black because there is no plane. We don't want to generate such useless masks and pollute our training data.

The solution is to only load the images that contains the classes we concern.

![](C:\Users\zhaoc\Documents\git\segmentation_models\img\VOCoverview.png)



#### construct VOC dataset

VOC介绍: https://arleyzhang.github.io/articles/1dc20586/ 

三种常用数据集：https://zhuanlan.zhihu.com/p/48670341



#### construct the seg model

```
import segmentation_models as sm
```

+ set the CONSTANTS

  BACKBONE = 'efficientnetb3'
  BATCH_SIZE = 8
  CLASSES = ['car']
  LR = 0.0001
  EPOCHS = 40

  preprocess_input = sm.get_preprocessing(BACKBONE)

+ create model

```python
model = \
segmentation_models.Unet(backbone_name='vgg16', 
                         input_shape=(None, None, 3), #image of any size, 3 channels
                         classes=1, 				#output classes number
                         activation='sigmoid', 		#set to 'softmax' if multiple classes
                         weights=None, 				#
                         encoder_weights='imagenet', 
                         #pretrained on on 2012 ILSVRC ImageNet dataset
                         encoder_freeze=False, 		#set to Ture cuz Transfer Learning
                         encoder_features='default', #layer list of skip connection
                         decoder_block_type='upsampling', 
                         decoder_filters=(256, 128, 64, 32, 16), 
                         decoder_use_batchnorm=True, 
                         **kwargs)
```

#### compile the model

```python
optim = keras.optimizers.Adam(LR)
dice_loss = sm.losses.DiceLoss()
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
```

```python
model.compile(optimizer, loss, metrics)
```

