#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import glob
import shutil


# In[2]:


# Defining the path for train and test images
## Todo: Update the paths of the train and test dataset

data_train=pathlib.Path('C:/Users/Admin/Downloads/CNN_assignment/Train')
data_test=pathlib.Path('C:/Users/Admin/Downloads/CNN_assignment/Test')


# In[3]:


# image count of train and test data

img_count_train=len(list(data_train.glob('*/*.jpg')))
print(img_count_train)
img_count_test=len(list(data_test.glob('*/*.jpg')))
print(img_count_test)


# In[4]:


batch_size = 32
img_height = 180
img_width = 180


# In[5]:


# Writing train dataset 
# using seed=123 while creating dataset using tf.keras.preprocessing.image_dataset_from_directory
# resizing images to the size img_height*img_width, while writting the dataset

train_data=tf.keras.preprocessing.image_dataset_from_directory(data_train,seed=123,validation_split=0.2,subset='training',image_size=(img_height,img_width),batch_size=batch_size)


# In[6]:


## Writing validation dataset 
# using seed=123 while creating dataset using tf.keras.preprocessing.image_dataset_from_directory
# resizing images to the size img_height*img_width, while writting the dataset

val_data=tf.keras.preprocessing.image_dataset_from_directory(data_train,seed=123,validation_split=0.2,subset='validation', image_size=(img_height,img_width),batch_size=batch_size)


# In[7]:


# Listing out all the classes of skin cancer and storing them in a list.

class_names=train_data.class_names
class_names


# In[8]:


type(train_data)


# In[9]:


for images,labels in train_data.take(1):
  print(len(images))
  print(len(labels))


# ## Visualize the data

# In[10]:


plt.figure(figsize=(10,10))
for images, labels in train_data.take(1):
  print(len(images))
  print(len(labels))
  plt.imshow(images[10].numpy().astype("uint8"))
  plt.title(class_names[labels[10]])
  plt.axis("off")


# In[11]:


plt.figure(figsize=(10,10))
for images,labels in train_data.take(5):
  for i in range(9):
    ax=plt.subplot(3,3,i+1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


# In[12]:


print(type(train_data))
print(len(train_data))


# In[13]:


# overlaps data preprocessing and model execution while training.,Speeding up training

AUTOTUNE=tf.data.experimental.AUTOTUNE
train_data=train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_data=val_data.cache().prefetch(buffer_size=AUTOTUNE)


# ## creating model

# In[14]:


num_classes=9

#A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor
#2D convolution layer (e.g. spatial convolution over images).We slide over the feature map and extract tiles of a specified size.
# Downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window (of size defined by pool_size) for each channel of the input.
#Advantages of downsampling - Decreased size of input for upcoming layers, Works against overfitting. Flattens all its structure to create a single long feature vector

model=Sequential([layers.experimental.preprocessing.Rescaling(1./255,input_shape=(img_height,img_width,3)),
                  layers.Conv2D(16,3,padding='same',activation='relu'),layers.MaxPooling2D(),
                  layers.Conv2D(32,3,padding='same',activation='relu'),layers.MaxPooling2D(),layers.Conv2D(64,3,padding='same',activation='relu'),
                  layers.MaxPooling2D(),layers.Flatten(),layers.Dense(128,activation='relu'),layers.Dense(num_classes)])


# In[15]:


# View the summary of all layers

model.summary()


# In[16]:


# choosing an appropirate optimiser and loss function
# RMSprop. RMSprop is a very effective, but currently unpublished adaptive learning rate method
# Adam is a recently proposed update that looks a bit like RMSProp with momentum. 

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])


# In[17]:


epochs=20
history=model.fit(train_data,validation_data=val_data,epochs=epochs)


# # Training the model

# In[18]:


accu=history.history['accuracy']
val_accu=history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range=range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1,2,1)
plt.plot(epochs_range,accu,label='Training Accuracy')
plt.plot(epochs_range,val_accu,label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range,loss,label='Training Loss')
plt.plot(epochs_range,val_loss,label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[19]:


loss,accu=model.evaluate(train_data,verbose=1,)
val_loss,val_accu=model.evaluate(val_data,verbose=1)

print("Accuracy:",accu)
print("Validation Accuracy:",val_accu)
print("Loss:",loss)
print("Validation Loss",val_loss)


# In[20]:


# again modelling

model=Sequential([layers.experimental.preprocessing.Rescaling(1./255,input_shape=(img_height,img_width,3)),
                  layers.Conv2D(16,3,padding='same',activation='relu'),layers.MaxPooling2D(),layers.Conv2D(32,3,padding='same',activation='relu'),
                  layers.MaxPooling2D(),layers.Conv2D(64,3,padding='same',activation='relu'),layers.MaxPooling2D(),
                  layers.Flatten(),layers.Dense(128, activation='relu'),  layers.Dense(num_classes)])


# ## visualizing training data

# In[21]:


# after analyseing the model fit history for presence of underfit or overfit, choosing an appropriate data augumentation strategy.

data_augmentation=keras.Sequential([layers.experimental.preprocessing.RandomFlip("horizontal",input_shape=(img_height,img_width,3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.RandomTranslation(1,.5,fill_mode="reflect",interpolation="bilinear",seed=None,fill_value=0.0),
    layers.experimental.preprocessing.RandomCrop(img_height,img_width),])


# In[22]:


plt.figure(figsize=(10,10))
for images,_ in train_data.take(1):
  for i in range(9):
    augmented_images=data_augmentation(images)
    ax=plt.subplot(3,3,i+1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")


# ### create the model compile and train the model

# In[23]:


model=Sequential([data_augmentation,layers.experimental.preprocessing.Rescaling(1./255),layers.Conv2D(64,3,padding='same',activation='relu'),
                  layers.MaxPooling2D(),layers.Conv2D(128,3,padding='same', activation='relu'),layers.MaxPooling2D(),
                  layers.Conv2D(256,3,padding='same',activation='relu'),layers.MaxPooling2D(),layers.Dropout(0.2),
                  layers.Flatten(),layers.Dense(128,activation='relu'),layers.Dense(num_classes)])


# In[24]:


model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])


# In[25]:


epochs=20
history=model.fit(train_data,validation_data=val_data,epochs=epochs)


# ## visualizing the results

# In[26]:


accu=history.history['accuracy']
val_accu=history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range=range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1,2,1)
plt.plot(epochs_range,accu,label='Training Accuracy')
plt.plot(epochs_range,val_accu,label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range,loss,label='Training Loss')
plt.plot(epochs_range,val_loss,label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[27]:


loss,accu=model.evaluate(train_data,verbose=1,)
val_loss,val_accu=model.evaluate(val_data,verbose=1)

print("Accuracy:",accu)
print("Validation Accuracy:",val_accu)
print("Loss:",loss)
print("Validation Loss",val_loss)


# In[28]:


# For convenience, let us set up the path for the training and validation sets

train_dir=os.path.join('C:/Users/Admin/Downloads/CNN_assignment/Train')
val_dir=os.path.join('C:/Users/Admin/Downloads/CNN_assignment/Test')


# In[29]:


# Setting batch size and image size

batch_size=100
IMG_SHAPE=224

# Create training images generator
# Generate batches of tensor image data with real-time data augmentation.

image_gen_train=ImageDataGenerator(rescale=1./255,rotation_range=45,width_shift_range=.15,height_shift_range=.15,
                                   horizontal_flip=True,zoom_range=0.5)

#Then calling image_dataset_from_directory(main_directory, labels='inferred') will return a tf.data.Dataset that yields batches of images from the subdirectories

train_data_gen=image_gen_train.flow_from_directory(batch_size=batch_size,directory=train_dir,shuffle=True,
                                                     target_size=(IMG_SHAPE,IMG_SHAPE),class_mode='sparse')

# Creating validation images generator

image_gen_val=ImageDataGenerator(rescale=1./255)
val_data_gen=image_gen_val.flow_from_directory(batch_size=batch_size,directory=val_dir,target_size=(IMG_SHAPE, IMG_SHAPE),
                                               class_mode='sparse')


# In[30]:


# Create a CNN model
# A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor

model = Sequential()

#2D convolution layer (e.g. spatial convolution over images).

model.add(Conv2D(16,3,padding='same',activation='relu',input_shape=(IMG_SHAPE,IMG_SHAPE,3)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32,3,padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(9))


# In[31]:


# Compile the model

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])


# In[32]:


# Train the model

epochs = 20
history = model.fit(train_data_gen,validation_data=val_data_gen,epochs=10)


# In[33]:


#visualizing the data

epochs=10
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range=range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range,acc,label='Training Accuracy')
plt.plot(epochs_range,val_acc,label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range,loss,label='Training Loss')
plt.plot(epochs_range,val_loss,label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[34]:


# comparing with previous model

from glob import glob

path_list=[x for x in glob(os.path.join(data_train,'*','*.jpg'))]
lesion_list=[os.path.basename(os.path.dirname(y)) for y in glob(os.path.join(data_train,'*','*.jpg'))]
len(path_list)


# In[35]:


dataframe_dict_original=dict(zip(path_list,lesion_list))
original_df=pd.DataFrame(list(dataframe_dict_original.items()),columns=['Path','Label'])
original_df


# In[36]:


from sklearn.preprocessing import LabelEncoder
from collections import Counter

# split into input and output elements

X,y=original_df['Path'],original_df['Label']

y=LabelEncoder().fit_transform(y)

counter=Counter(y)
for k,v in counter.items():
    per=v/len(y)*100
    print('Class=%d,n=%d(%.3f%%)'%(k,v,per))

plt.bar(counter.keys(), counter.values())
plt.show()


# In[37]:


datapath='C:/Users/Admin/Downloads/CNN_assignment/Train/actinic keratosis'

import Augmentor

p=Augmentor.Pipeline(datapath)

p.rotate(probability=0.7,max_left_rotation=10,max_right_rotation=10)
p.zoom(probability=0.5,min_factor=1.1,max_factor=1.5)
p.sample(150)
p.process()


# In[38]:


image_count_train=len(list(data_train.glob('*/output/*.jpg')))
print(image_count_train)


# In[39]:


path_list_new=[x for x in glob(os.path.join(data_train,'*','output','*.jpg'))]
path_list_new


# In[40]:


lesion_list_new=[os.path.basename(os.path.dirname(os.path.dirname(y))) for y in glob(os.path.join(data_train,'*','output','*.jpg'))]
lesion_list_new


# In[41]:


dataframe_dict_new=dict(zip(path_list_new,lesion_list_new))


# In[42]:


df2=pd.DataFrame(list(dataframe_dict_new.items()),columns=['Path','Label'])
new_df=original_df.append(df2)


# In[43]:


new_df['Label'].value_counts()


# ## Train the model on the data 

# In[44]:


train_ds=tf.keras.preprocessing.image_dataset_from_directory(data_train,seed=123,validation_split=0.2,subset='training',
                                                             image_size=(img_height,img_width),batch_size=batch_size)


# In[45]:


val_ds=tf.keras.preprocessing.image_dataset_from_directory(data_train,seed=123,validation_split=0.2,subset='validation',
                                                           image_size=(img_height,img_width),batch_size=batch_size)


# ## creating model

# In[46]:


AUTOTUNE=tf.data.experimental.AUTOTUNE

train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds=val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model=Sequential([layers.experimental.preprocessing.Rescaling(1./255),layers.Conv2D(16,3,padding='same',activation='relu'),
                  layers.MaxPooling2D(),layers.Conv2D(32,3,padding='same',activation='relu'),layers.MaxPooling2D(),
                  layers.Conv2D(64,3,padding='same',activation='relu'),layers.MaxPooling2D(),layers.Dropout(0.2),
                  layers.Flatten(),layers.Dense(128, activation='relu'),layers.Dense(num_classes)])


# In[47]:


#compile model

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])


# In[48]:


# training the model

epochs=20
history=model.fit(train_ds,validation_data=val_ds,epochs=epochs)


# In[49]:


# visualing the data

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range=range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range,acc,label='Training Accuracy')
plt.plot(epochs_range,val_acc,label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range,loss,label='Training Loss')
plt.plot(epochs_range,val_loss,label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Accuracy has been increased on train data by using Augmentor library
