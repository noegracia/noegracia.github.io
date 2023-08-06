---
title: "Skin Cancer Detection using semantic segmentation"
date: 2021-08-10T08:13:25+01:00
description: Personal project to learn more about neural networks and  Semantic segmentation
menu:
  sidebar:
    name: Semantic Segmentation
    identifier: semantic_segmentation_for_cancer
    parent: ml
    weight: 10
hero: images/portada-segm.jpg
tags: ["AI","ML","Autoencoder","Personal Project"]
categories: ["Basic"]
---

The objective of this project is to build a Skin Cancer Detection Tool. The tool that we are creating is a segmentation model of spots (moles, melanomas, etc...) on microscopic images of the skin. To create this tool we will have to train a semantic segmentation AI model. The data that we use for that training is from [The International Skin Imaging Collaboration](https://gallery.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery?filter=%5B%5D).

# Segmentation model

## Imports
```python
from locale import normalize
import tensorflow as tf
from tensorflow.keras.layers import *
import os
import pathlib
import cv2
import numpy as np
import randomsklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
```

## Hyperparameters and reproductibility
```python
IMG_WIDTH = 128
IMG_HEIGHT= 128
IMG_CHANNELS = 3

BATCH_SIZE = 16
EPOCHS = 80
SEED = 123

TRAIN_PATH = 'train/'
TRAIN_LABEL_PATH = 'train_labels/'
```

## Loading the data

```python
def get_data():
    print("Loading data...")
    x_train = []
    y_train = []

    # Load in the images
    for filepath in os.listdir(TRAIN_PATH):
        x_img = cv2.imread(TRAIN_PATH+filepath)
        x_img = cv2.resize(x_img, (IMG_HEIGHT,IMG_WIDTH)) # resize to fit in UNET model
        # x_img = tf.cast(x_img, tf.float32) / 255.0 # normalize -> return: tensor of floats [0-1]
        x_train.append(x_img)
    x_train = np.array(x_train).reshape(-1, IMG_HEIGHT,IMG_WIDTH,3)

    for filepath in os.listdir(TRAIN_LABEL_PATH):
        y_img = cv2.imread(TRAIN_LABEL_PATH+filepath,cv2.IMREAD_GRAYSCALE)
        y_img = cv2.resize(y_img, (IMG_HEIGHT,IMG_WIDTH)) # resize to fit in UNET model
        # y_img = tf.cast(y_img, tf.float32) / 255.0 # normalize -> return: tensor of floats [0-1]
        y_img = y_img==255
        y_img = np.expand_dims(y_img,axis=-1)
        y_train.append(y_img)
    y_train = np.array(y_train).reshape(-1, IMG_HEIGHT,IMG_WIDTH,1)
    print("Data loaded!")
    return x_train,y_train
```

## U-NET IMPLEMENTATION

The solution we are going to implement is a U-NET neural network.

```python
def build_model(input_size, start_neurons, initializer):
    print("Building UNET model")
    
    inputs = Input(input_size)
    # normalisation de l'input
    input_layer = Lambda(lambda x: x / 255)(inputs)

    # encoder
    conv_1 = Conv2D(start_neurons * 1, 3, activation='relu', kernel_initializer=initializer, padding='same')(input_layer)
    conv_1 = Conv2D(start_neurons * 1, 3, activation='relu', kernel_initializer=initializer, padding='same')(conv_1)

    maxpool_2 = MaxPooling2D(2)(conv_1)
    maxpool_2 = Dropout(0.2)(maxpool_2) #  dropping 20% data to avoid overfitting
    conv_2 = Conv2D(start_neurons * 2, 3, activation='relu', kernel_initializer=initializer, padding='same')(maxpool_2)
    conv_2 = Conv2D(start_neurons * 2, 3, activation='relu', kernel_initializer=initializer, padding='same')(conv_2)

    maxpool_3 = MaxPooling2D(2)(conv_2)
    maxpool_3 = Dropout(0.4)(maxpool_3)
    conv_3 = Conv2D(start_neurons * 4, 3, activation='relu', kernel_initializer=initializer, padding='same')(maxpool_3)
    conv_3 = Conv2D(start_neurons * 4, 3, activation='relu', kernel_initializer=initializer, padding='same')(conv_3)

    maxpool_4 = MaxPooling2D(2)(conv_3)
    maxpool_4 = Dropout(0.4)(maxpool_4)
    conv_4 = Conv2D(start_neurons * 8, 3, activation='relu', kernel_initializer=initializer, padding='same')(maxpool_4)
    conv_4 = Conv2D(start_neurons * 8, 3, activation='relu', kernel_initializer=initializer, padding='same')(conv_4)

    # bottom
    maxpool_bottom = MaxPooling2D(2)(conv_4)
    maxpool_bottom = Dropout(0.4)(maxpool_bottom)
    conv_bottom = Conv2D(start_neurons * 16, 3, activation='relu', kernel_initializer=initializer, padding='same')(maxpool_bottom)
    conv_bottom = Conv2D(start_neurons * 16, 3, activation='relu', kernel_initializer=initializer, padding='same')(conv_bottom)

    # decoder

    upconv_4 = Conv2DTranspose(start_neurons * 8, 3, strides=2, padding="same")(conv_bottom)
    merge_4 = concatenate([upconv_4, conv_4])
    merge_4 = Dropout(0.5)(merge_4)
    deconv_4 = Conv2D(start_neurons * 8, 3, activation="relu", padding="same")(merge_4)
    deconv_4 = Conv2D(start_neurons * 8, 3, activation="relu", padding="same")(deconv_4)

    upconv_3 = Conv2DTranspose(start_neurons * 4, 3, strides=2, padding="same")(deconv_4)
    merge_3 = concatenate([upconv_3, conv_3])
    merge_3 = Dropout(0.5)(merge_3)
    deconv_3 = Conv2D(start_neurons * 4, 3, activation="relu", padding="same")(merge_3)
    deconv_3 = Conv2D(start_neurons * 4, 3, activation="relu", padding="same")(deconv_3)

    upconv_2 = Conv2DTranspose(start_neurons * 2, 3, strides=2, padding="same")(deconv_3)
    merge_2 = concatenate([upconv_2, conv_2])
    merge_2 = Dropout(0.5)(merge_2)
    deconv_2 = Conv2D(start_neurons * 2, 3, activation="relu", padding="same")(merge_2)
    deconv_2 = Conv2D(start_neurons * 2, 3, activation="relu", padding="same")(deconv_2)

    upconv_1 = Conv2DTranspose(start_neurons * 1, 3, strides=2, padding="same")(deconv_2)
    merge_1 = concatenate([upconv_1, conv_1])
    merge_1 = Dropout(0.5)(merge_1)
    deconv_1 = Conv2D(start_neurons * 1, 3, activation="relu", padding="same")(merge_1)
    deconv_1 = Conv2D(start_neurons * 1, 3, activation="relu", padding="same")(deconv_1)

    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(deconv_1)
    
    print(input_layer.shape,output_layer.shape)
    input("---")

    model = tf.keras.Model(inputs = input_layer, outputs = output_layer)
    print("UNET built!")
    
    return model
```

## Building the model

```python
initializer = 'he_normal'
input_size = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

model = build_model(input_size, 16, initializer)
```

## Compiling the model
```python
model.compile(optimizer='adam', loss = tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
# model.summary()
```

## Model checkpoint
```python
checkpointer = tf.keras.callbacks.ModelCheckpoint('seg_lesions_cut.h5', verbose=1, save_best_only=True)

file_name  =  'my_saved_model_80'

callbacks  = [
    checkpointer,
    tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs\\{}'.format(file_name),)
]
```

# Inference

## Imports

```python
import cv2
import tensorflow as tf
import numpy as np
import os
```

## Hyperparameters and reproductibility
```python
INPUT_FOLDER = 'Input_folder/'
OUTPUT_FOLDER = 'Output_folder/'

IMG_WIDTH = 128
IMG_HEIGHT= 128
IMG_CHANNELS = 3
```


## Load data

```python
def get_data():
    print("Loading data...")
    x_train = []

    # Load in the images
    for filepath in os.listdir(INPUT_FOLDER):
        x_img = cv2.imread(INPUT_FOLDER+filepath)
        x_img = cv2.resize(x_img, (IMG_HEIGHT,IMG_WIDTH)) # resize to fit in UNET model
        # x_img = tf.cast(x_img, tf.float32) / 255.0 # normalize -> return: tensor of floats [0-1]
        x_train.append(x_img)
    x_train = np.array(x_train).reshape(-1, IMG_HEIGHT,IMG_WIDTH,3)

    print("Data loaded!")
    return x_train
```

## Predict

```python
model = tf.keras.models.load_model("segm_model")

prediction = model.predict(get_data())

for i,img in enumerate(prediction):
    img = cv2.resize(img,(320,240))
    img = np.rint(img)
    cv2.imshow("test",img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    cv2.imwrite(OUTPUT_FOLDER+str(i)+".bmp",img)
```