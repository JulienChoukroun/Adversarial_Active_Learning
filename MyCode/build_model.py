# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 17:17:56 2017

@author: mducoffe
"""

##### These comments are from Julien Choukroun
##### Now we use tensorflow.keras instead of keras because we are in Tensorflow 2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, InputLayer

def build_model_AlexNet(img_size, nb_classes):
    
    nb_pool = 2
    model = Sequential()
     
    nb_channel, img_rows, img_cols = img_size
    	#layer 1
    model.add(Conv2D(96, (11, 11), padding='same', input_shape = (nb_channel, img_rows, img_cols), data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
    model.add(Dropout(0.25))
    
    #layer 2
    model.add(Conv2D(256, (5, 5), padding='same', data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
    model.add(Dropout(0.25))
    
    #layer 3
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), padding='same', data_format='channels_first'))
    model.add(Activation('relu'))
    
    #layer 4
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(1024, (3, 3), padding='same', data_format='channels_first'))
    model.add(Activation('relu'))
    
    #layer 5
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(1024, (3, 3), padding='same', data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
    model.add(Dropout(0.25))
    
    #layer 6
    model.add(Flatten())
    model.add(Dense(3072, kernel_initializer="glorot_normal"))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    #layer 7
    model.add(Dense(4096, kernel_initializer="glorot_normal"))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    #layer 8
    model.add(Dense(nb_classes, kernel_initializer="glorot_normal"))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["acc"])
    
    return model;

##### I use VGG8   
def build_model_VGG8(img_size, nb_classes):
    
    nb_conv = 3
    nb_pool = 2
    nb_channel, img_rows, img_cols = img_size

    model = Sequential()
    ##### The order has change, now the channel is the last parameter instead of the first parameter
    model.add(ZeroPadding2D((1,1),input_shape=(img_rows,img_cols,nb_channel)))
    ##### I change the data_format='channels_first' by 'channels_last'
    model.add(Conv2D(64, (nb_conv, nb_conv), activation='relu', data_format='channels_last'))
    model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(ZeroPadding2D((1,1)))
    ##### I change the data_format='channels_first' by 'channels_last'
    model.add(Conv2D(128, (nb_conv, nb_conv), activation='relu', data_format='channels_last'))
    model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(ZeroPadding2D((1,1)))
    ##### I change the data_format='channels_first' by 'channels_last'
    model.add(Conv2D(256, (nb_conv, nb_conv), activation='relu', data_format='channels_last'))
    model.add(ZeroPadding2D((1,1)))
    ##### I change the data_format='channels_first' by 'channels_last'
    model.add(Conv2D(256, (nb_conv, nb_conv), activation='relu', data_format='channels_last'))
    model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(ZeroPadding2D((1,1)))
    ##### I change the data_format='channels_first' by 'channels_last'
    model.add(Conv2D(512, (nb_conv, nb_conv), activation='relu', data_format='channels_last'))
    model.add(ZeroPadding2D((1,1)))
    ##### I change the data_format='channels_first' by 'channels_last'
    model.add(Conv2D(512, (nb_conv, nb_conv), activation='relu', data_format='channels_last'))
    model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["acc"])
     
    return model

##### I use LeNet5     
def build_model_LeNet5(img_size, nb_classes):

    nb_pool = 2
    nb_channel, img_rows, img_cols = img_size
    model = Sequential()
    
    ##### I change the data_format='channels_first' by 'channels_last'
    model.add(Conv2D(6, (5, 5), padding='valid', input_shape = (img_rows, img_cols, nb_channel), data_format='channels_last'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
    model.add(Dropout(0.25))
    
    ##### I change the data_format='channels_first' by 'channels_last'
    model.add(Conv2D(16, (5, 5), padding='valid', data_format='channels_last'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((nb_pool,nb_pool), strides=(2,2)))
    model.add(Dropout(0.25))
    
    ##### I change the data_format='channels_first' by 'channels_last'
    model.add(Conv2D(120, (1, 1), padding='valid', data_format='channels_last'))
    
    model.add(Flatten())
    model.add(Dense(84, activation='relu'))
    model.add(Dense(nb_classes))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["acc"])
     
    return model
     
def build_model_func(network_archi, img_size=(1,28,28, 10)):
    
    network_archi = network_archi.lower()
    num_classes = img_size[3]
    img_size = (img_size[0], img_size[1], img_size[2])
    model = None
    assert (network_archi in ['vgg8', 'lenet5', 'alexnet']), ('unknown architecture', network_archi)
    if network_archi == 'vgg8':
        model = build_model_VGG8(img_size, nb_classes=num_classes)
    if network_archi == 'lenet5':
        model = build_model_LeNet5(img_size,nb_classes=num_classes)
    if network_archi == 'alexnet':
        model = build_model_AlexNet(img_size, nb_classes=num_classes)
        
    return model

