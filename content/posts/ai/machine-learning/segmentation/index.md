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

# Skin Cancer Detection Tool README.md

## Overview

The objective of this project is to build a Skin Cancer Detection Tool. The tool that we are creating is a segmentation model of spots (moles, melanomas, etc...) on microscopic images of the skin. To create this tool we will have to train a semantic segmentation AI model. The data that we use for that training is from [The International Skin Imaging Collaboration](https://gallery.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery?filter=%5B%5D).

## File Descriptions:

1. **data.py**: Contains functions to process and load the dataset, preprocess the images, and masks and to create TensorFlow datasets.
    - `process_data(data_path, file_path)`: Reads the image and mask paths from the dataset.
    - `load_data(path)`: Load training, validation, and test data.
    - `read_image(x)` and `read_mask(x)`: Read the images and the masks respectively.
    - `tf_dataset(x, y, batch=8)`: Create a TensorFlow dataset.
    - `preprocess(x, y)`: Preprocess the images and masks.

2. **predict.py**: Uses a pretrained model to make predictions on new images.
    - `get_data()`: Load test images from the `INPUT_FOLDER`.
    - Then, predictions are made using the loaded model and saved to the `OUTPUT_FOLDER`.

## Setup & Requirements

### Requirements:
* python 3.x
* pandas
* numpy
* scikit-learn
* tensorflow 2.x
* opencv-python

You can install these requirements using:
```bash
pip install pandas numpy scikit-learn tensorflow opencv-python
```

### Steps to Run:

1. **Data Preparation**:
   * Place your dataset in an appropriate directory.
   * Adjust the paths in the `data.py` script.
   * Run the `data.py` script to check if data is loaded properly.
     ```bash
     python data.py
     ```

2. **Predicting**:
   * Place your test images in the `INPUT_FOLDER`.
   * Ensure the model path "segm_model" in `predict.py` corresponds to your trained model.
   * Run the `predict.py` script to make predictions.
     ```bash
     python predict.py
     ```

## Notes

- This tool currently segments the spots and saves the segmented images in the `OUTPUT_FOLDER`.
- You might need to train the model first using your data to get the "segm_model".
- Ensure the directories mentioned in the scripts exist or are modified according to your directory structure.