---
title: "Autoencoder"
date: 2022-01-10T08:11:25+01:00
description: An autoencoder is a type of artificial neural network used to learn efficient codings of unlabeled data.
menu:
  sidebar:
    name: Autoencoder
    identifier: autoencoder
    parent: ml
    weight: 10
hero: images/coded-decoded-mnist.jpg
tags: ["AI","ML","Autoencoder"]
categories: ["Basic"]
---

Creation of pre-trained autoencoder to learn the initial condensed representation of unlabeled datasets. This architecture consists of 3 parts:

1. Encoder: Compresses the input data from the train-validation-test set into a coded representation which is typically smaller by several orders of magnitude than the input data.
2. Latent Space: This space contains the compressed knowledge representations and is thus the most crucial part of the network.
3. Decoder: A module that helps the network to "decompress" the knowledge representations and reconstruct the data from their coded form. The output is then compared to a ground truth.

## Imports
```python
from time import time
import numpy as np
import keras.backend as K
from keras.layers import Dense, Input, Layer, InputSpec,  Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Conv2DTranspose
from keras.models import Model
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
```

## Loading the data

```python
from keras.datasets import mnist
from keras.datasets import fashion_mnist

import numpy as np

# Chargement et normalisation (entre 0 et 1) des données de la base de données MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train), 784))
x_test = np.reshape(x_test, (len(x_test), 784))
```

## Classic Autoencoder

```python
# Dimension de l'entrée
input_img = Input(shape=(784,))

# Dimension de l'espace latent : PARAMETRE A TESTER !!
latent_dim = 10

# Définition du encodeur
x0 = Dense(500, activation='relu')(input_img)
x = Dense(200, activation='relu')(x0)
encoded = Dense(latent_dim, activation='relu')(x)


# Définition du décodeur
decoder_input = Input(shape=(latent_dim,))
x = Dense(200, activation='relu')(decoder_input)
x1 = Dense(500, activation='relu')(x)
decoded = Dense(784, activation='relu')(x1)

# Construction d'un modèle séparé pour pouvoir accéder aux décodeur et encodeur
encoder = Model(input_img, encoded)
decoder = Model(decoder_input, decoded)


# Construction du modèle de l'auto-encodeur
encoded = encoder(input_img)
decoded = decoder(encoded)
autoencoder = Model(input_img, decoded)
```

## Summary
```python
# Autoencodeur 
autoencoder.compile(optimizer='Adam', loss='mse')
autoencoder.summary()
print(encoder.summary())
print(decoder.summary())
```

## Training

```python
autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))
```

## Evaluation

```python
# Encode and decode some digits
# Note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
```

## Visualization

```python
n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

{{<img src="/posts/ai/machine-learning/autoencoder/images/coded-decoded-mnist.png" align="center" title="Coded-Decoded-MNIST">}}

{{< vs >}}

## Display

```python
# Affichage
count=1000
idx = np.random.choice(len(x_test), count)
inputs = x_test[idx]

coordsAC = encoder.predict(inputs)
coordsTSNE = TSNE(n_components=2).fit_transform(inputs.reshape(count, -1))
coordsPCA = PCA(n_components=2).fit_transform(inputs.reshape(count, -1))



classes = y_test[idx]

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title("Espace latent")
plt.scatter(coordsAC[:, 0], coordsAC[:, 1], c=classes, cmap="Paired")
plt.colorbar()

fig2, ax2 = plt.subplots(figsize=(10, 7))
ax2.set_title("ACP sur espace latent")
plt.scatter(coordsPCA[:, 0], coordsPCA[:, 1], c=classes, cmap="Paired")
plt.colorbar()



fig3, ax3 = plt.subplots(figsize=(10, 7))
ax3.set_title("tSNE sur espace latent")
plt.scatter(coordsTSNE[:, 0], coordsTSNE[:, 1], c=classes, cmap="Paired")
plt.colorbar()
```
{{<img src="/posts/ai/machine-learning/autoencoder/images/latent-space.png" align="center" title="Coded-Decoded-MNIST">}}

{{< vs >}}

{{<img src="/posts/ai/machine-learning/autoencoder/images/latent-space-acp.png" align="center" title="Coded-Decoded-MNIST">}}

{{< vs >}}

{{<img src="/posts/ai/machine-learning/autoencoder/images/latent-space-tsne.png" align="center" title="Coded-Decoded-MNIST">}}

{{< vs >}}
