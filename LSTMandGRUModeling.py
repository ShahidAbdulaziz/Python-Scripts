#!/usr/bin/env python
# coding: utf-8

# # DSCI - 619 Final Project
# **Shahid Abdulaziz**

# In[9]:


from zipfile import ZipFile
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from numpy import array
from numpy import argmax
from ipywidgets import IntProgress
import tensorflow_datasets as tfds
import seaborn as sns
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import datasets, layers, models
from collections import Counter
import matplotlib.pyplot as plt
import os
import re
from sklearn.metrics import classification_report


# # Part 1

# ## Problem :

# In[10]:


with ZipFile("archive.zip", "r") as zipobj:
        zipobj.extractall()        


# ## Problem 2:

# In[11]:


data_dir  = 'C:\\Users\\shahi\\downloads\\indoorCVPR_09\\Images'
classes = os.listdir(data_dir)


# In[12]:


batch_size = 32
image_height = 256
image_width = 256
train_test_split = 0.2


train_data = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  labels='inferred', # labels are generated from the directory structure
  label_mode= 'int', #int: means that the labels are encoded as integers (e.g. for sparse_categorical_crossentropy loss).
  validation_split= train_test_split,
  subset="training",
  seed= 1001, #fix the seed
  image_size=(image_height, image_width),
  batch_size=batch_size)


# In[13]:


val_data = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  labels='inferred',
  label_mode='int',
  validation_split= train_test_split,
  subset="validation",
  seed=1001,
  image_size=(image_height, image_width),
  batch_size=batch_size,
)


# In[14]:



num_labels = len(classes)
print(f'There are {num_labels} classes in the image dataset')
image_channel = 3
print(f' There are {image_channel} channels in the images')


# In[15]:


normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1.0/255.0)

AUTOTUNE = tf.data.AUTOTUNE # Tune the value dynamically at runtime.
train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)


# In[16]:


model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(image_height, image_width, image_channel)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dense(num_labels)
])


# In[17]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[18]:


get_ipython().run_cell_magic('time', '', "callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience= 3)\nepochs= 3\nhistory = model.fit(\n  train_data,\n  validation_data=val_data,\n  epochs=epochs,\n  callbacks=[callback], verbose = 1\n)")


# In[19]:


train_history = pd.DataFrame(history.history)
train_history['epoch'] = history.epoch
sns.lineplot(x='epoch', y ='loss', data =train_history)
sns.lineplot(x='epoch', y ='val_loss', data =train_history)
plt.legend(labels=['train_loss', 'val_loss'])


# In[20]:


sns.lineplot(x='epoch', y ='accuracy', data =train_history)
sns.lineplot(x='epoch', y ='val_accuracy', data =train_history)
plt.legend(labels=['train_accuracy', 'val_accuracy'])


# ## Problem 3:

# In[303]:


data_aug = tf.keras.Sequential(
  [
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", 
                                                 input_shape=(image_height, 
                                                              image_width,
                                                              image_channel)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1,width_factor = 0.1 ),
    tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(0.1, 0.1))
  ]
)


# In[304]:


model = tf.keras.Sequential([
  data_aug,
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(num_labels)
])


# In[305]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[306]:


get_ipython().run_cell_magic('time', '', "callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience= 3)\nepochs= 3\nhistory = model.fit(\n  train_data,\n  validation_data=val_data,\n  epochs=epochs,\n  callbacks=[callback], verbose = 1\n)")


# In[308]:


train_history = pd.DataFrame(history.history)
train_history['epoch'] = history.epoch
sns.lineplot(x='epoch', y ='loss', data =train_history)
sns.lineplot(x='epoch', y ='val_loss', data =train_history)
plt.legend(labels=['train_loss', 'val_loss'])


# In[309]:


sns.lineplot(x='epoch', y ='accuracy', data =train_history)
sns.lineplot(x='epoch', y ='val_accuracy', data =train_history)
plt.legend(labels=['train_accuracy', 'val_accuracy'])


# ## Problem 4:

# In[21]:


IMG_SHAPE = (image_height, image_width, image_channel)
print(IMG_SHAPE )
MobileNetV3Large_model = tf.keras.applications.MobileNetV3Large(input_shape = IMG_SHAPE,
                                               include_top=False, # Remove the fully-connected layer
                                               weights='imagenet') # Pre-training on ImageNet


# In[22]:


MobileNetV3Large_model.trainable = False


# In[23]:


preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input


# In[24]:


image_batch, label_batch = next(iter(train_data))
feature_batch = MobileNetV3Large_model(image_batch)
print(feature_batch.shape)


# In[25]:


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)


# In[26]:


global_average_layer = tf.keras.layers.Flatten()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)


# In[37]:


prediction_layer = tf.keras.layers.Dense(23)
prediction_batch = prediction_layer(feature_batch_average)
print(f' The size of the predicted value for a given batch = {prediction_batch.shape}')


# In[38]:


data_aug = tf.keras.Sequential(
  [ tf.keras.layers.experimental.preprocessing.Rescaling(1.0/255.0),
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", 
                                                 input_shape=(image_height, 
                                                              image_width,
                                                              image_channel)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1,width_factor = 0.1 ),
    tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(0.1, 0.1))
  ]
)


# In[39]:


outputs 


# In[40]:


IMG_SHAPE = (image_height, image_width,  image_channel)

inputs = tf.keras.Input(shape = IMG_SHAPE)
x = data_aug(inputs)
x = preprocess_input(x)
x = MobileNetV3Large_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


# In[45]:


learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate =learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[46]:


model.summary()


# In[47]:


tf.keras.utils.plot_model(model, show_shapes=True)


# In[48]:


get_ipython().run_cell_magic('time', '', "callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience= 5)\nepochs= 3\nhistory = model.fit(\n  train_data,\n  validation_data=val_data,\n  epochs=epochs,\n  callbacks=[callback], verbose = 1\n)")


# In[49]:


train_history = pd.DataFrame(history.history)
train_history['epoch'] = history.epoch
sns.lineplot(x='epoch', y ='loss', data =train_history)
sns.lineplot(x='epoch', y ='val_loss', data =train_history)
plt.legend(labels=['train_loss', 'val_loss'])


# In[50]:


sns.lineplot(x='epoch', y ='accuracy', data =train_history)
sns.lineplot(x='epoch', y ='val_accuracy', data =train_history)
plt.legend(labels=['train_accuracy', 'val_accuracy'])


# ## Problem 5:

# For a lot of the models, overfitting and underfitting seem to be a real issue. They are most likely do to the low epoch given. I would select the second model because it's training and validation accuracies showed normal behavior unlike the other two visualizations. It may have a case of overfitting but that is something we can adjust for later on. Additionally, the model itself is simple to explain and I was always taught that if two models are close to one another, always pick the similar one. This is because it is easier to explain to stakeholders.

# # Part 2

# ## Problem 1:

# In[5]:


with ZipFile("archive2.zip", "r") as zipobj:
        zipobj.extractall()       


# In[2]:


sentiment = pd.read_csv('training.1600000.processed.noemoticon.csv', header = None)
sentiment = sentiment.rename(columns = {0: 'target',
                  1: 'ids', 
                  2: 'date', 
                  3: 'flag', 
                  4: 'user',
                  5: 'text'})

sentiment


# In[3]:


sentiment = sentiment.dropna()
sentiment['reviews'] = sentiment['text']+ sentiment['date']
sentiment = sentiment.drop(['flag','ids','date','user','text'] ,axis = 1)
sentiment


# In[90]:


#sentiment['target'] = sentiment['target'].astype(int)
#sentiment


# In[4]:


sentiment['reviews'] = sentiment['reviews'].apply(str).apply(lambda x: re.sub(r'\b[a-zA-Z]{1,2}\b', '', x))
sentiment['reviews'] = sentiment['reviews'].apply(lambda x: re.sub(r"", " ", x))
sentiment['reviews'] = sentiment['reviews'].apply(str).apply(lambda x: re.sub(r'[^A-Za-z0-9]+',' ',x))


# In[109]:


#sentiment['target'] = sentiment['target'].astype(str)


# In[5]:


features = sentiment['reviews'].values
labels = sentiment['target'].astype('category').cat.codes
X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels , test_size=0.2)


# In[6]:


VOCAB_SIZE = 1000
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(X_train)


#  ## Problem 3:

# In[337]:


model = tf.keras.Sequential([
    # Convert review text to indices
    encoder, 
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    # 
    tf.keras.layers.GRU(128, return_sequences=False),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


# In[338]:


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


# In[341]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(x=X_train,y=y_train,batch_size= 1000,epochs=1,\n          validation_data=(X_test,y_test), verbose= 1\n          )')


# In[342]:


y_pred = (model.predict(X_test)> 0.5).astype(int)


# In[343]:


print(classification_report(y_test, y_pred))


# ## Problem 4:

# In[344]:


model = tf.keras.Sequential([
    # Convert review text to indices
    encoder, 
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    # Binary classifier
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


# In[345]:


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


# In[346]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(x=X_train,y=y_train,batch_size= 4000,epochs=1,\n          validation_data=(X_test,y_test), verbose= 1\n          )')


# In[347]:


y_pred = (model.predict(X_test)> 0.5).astype(int)


# In[348]:


print(classification_report(y_test, y_pred))


# ## Problem 5:

# In[349]:


model = tf.keras.Sequential([
    # Convert review text to indices
    encoder, 
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    # 
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    # Binary classifier
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


# In[350]:


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


# In[351]:


get_ipython().run_cell_magic('time', '', 'history = model.fit(x=X_train,y=y_train,batch_size= 1000,epochs=1,\n          validation_data=(X_test,y_test), verbose= 1\n          )')


# In[ ]:


y_pred = (model.predict(X_test)> 0.5).astype(int)


# In[ ]:


print(classification_report(y_test, y_pred))


# ## Problem 6:

# I would use the combined LSTM and GRU since it has the highest accuracy out of all the models. The models took extreme long to run and I had to fix something resulting in having to rerun the models at 4:00 a.m so I changed the epoch to 1. So this is based off of only 1 Epoch.     
