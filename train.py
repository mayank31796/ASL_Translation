# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 21:17:01 2018

@author: mayan
"""
import cv2
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from keras import optimizers
import matplotlib.pyplot as plt
import PIL.Image
from keras.preprocessing.image import ImageDataGenerator




    

def model():
    #x_train,y_train,x_test,y_test=getData()
    img = cv2.imread('.\Data\\Train\\A\\1.png')
    cv2.imshow("d",img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print(img.shape)
    x_shape,y_shape,z=img.shape
    
    
    
    model=Sequential()
    model.add(Conv2D(16,kernel_size=(3,3),activation='relu',input_shape=(200,200,3)))
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(16,kernel_size=(3,3),activation='relu'))
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(16,kernel_size=(3,3),activation='relu'))
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation= 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation= 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(25,activation='softmax'))
    model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ,'mse' ])
    
    
    
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory('.\Data\Train', target_size=(200, 200), batch_size=32, class_mode='categorical',shuffle=True,seed=42)

    validation_set = test_datagen.flow_from_directory('.\Data\Validation', target_size=(200, 200), batch_size=32, class_mode='categorical',shuffle=True,seed=42)
    
    testing_set= train_datagen.flow_from_directory('.\Data\Test', target_size=(200, 200), batch_size=32, class_mode='categorical',shuffle=False)
    history = model.fit_generator(
        training_set,
        steps_per_epoch=500,
        epochs=5,
        validation_data = validation_set,
        validation_steps = 2000
      )
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    #plotting MSE
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('Meas Squared Error')
    plt.ylabel('Error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    print("\n\n\nSaving Model....\n\n\n")
    model.save("model.hdf5")

    print("\n\nSaving weights....\n\n\n")
    model.save_weights("model_weights.hdf5")

    return

def main():
    model()
    
if __name__=="__main__":
    main()
    
    

