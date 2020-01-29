# ---------------
# Date: 10/06/2019
# Place: Biella/Torino/Ventimiglia
# Author: Vittorio Mazzia
# Project: Python in The Lab Project
# ---------------

# import some important libraries
import os
######################################################
# !! run this BEFORE importing TF or keras !!
# run code only on a single, particular GPU
######################################################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
"""
Environment Variable Syntax   Results
CUDA_VISIBLE_DEVICES=1      Only device 1 will be seen
CUDA_VISIBLE_DEVICES=0,1    Devices 0 and 1 will be visible
CUDA_VISIBLE_DEVICES=”0,1”  Same as above, quotation marks are optional
CUDA_VISIBLE_DEVICES=0,2,3  Devices 0, 2, 3 will be visible; device 1 is masked
CUDA_VISIBLE_DEVICES=""     None GPU, only CPU
"""
# choose an available GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

# configure the level of logging to INFO
logging.basicConfig(level=logging.INFO)


class Model(object):
    """
    A class used to load and, if needed, train new models.
    Using the available attributes is possible to train models only for digits or
    for 47 different characters (all uppercase letters and digits plus some lowercase letters).
    
    ...
    
    Attributes
    ----------
    mainDir: str
        path of the main folder of the project
    datasetObj: obj
        dataset object. It is used as source of data for new training
    dirBin: str
        path of the bin directory where all trained weights are stored
    char: bool
        if False a model for only digits recognition is loaded
    epochs: int
        how many iterations over the dataset are performed before concluding the training
    batchSize: int
        how many images are processed before updating weights values
    train: bool
        if True a foced training is performed even if already trained weights are available in the bin folder
    verbose: bool
        if True print some useful information
    
    Methods
    -------
    buildDNN(n_classes)
           Build a deep neural network object tailored for the specified number of classes
    trainModel()
           Train a new model with the specified attributes of the instantiated object
    loadModel()
           Try to load a model from the dirBin folder. If not found an automatic training is launched
    plotHistory(history)     
           Plot the loss and accuracy curves for the training dataset 
    
    """
    def __init__(self, mainDir, datasetObj, dirBin = 'bin', char=False, epochs = 20, batchSize = 128, train=False, verbose=False):
        """
        Parameters
        ----------
        mainDir: str
            path of the main folder of the project
        datasetObj: obj
            dataset object. It is used as source of data for new training
        dirBin: str
            path of the bin directory where all trained weights are stored
        char: bool
            if False a model for only digits recognition is loaded
        epochs: int
            how many iterations over the dataset are performed before concluding the training
        batchSize: int
            how many images are processed before updating weights values
        train: bool
            if True a foced training is performed even if already trained weights are available in the bin folder
        verbose: bool
            if True print some useful information
        """
        
        self.dirBin = os.path.join(mainDir, dirBin)
        self.verbose = verbose
        self.dataset = datasetObj
        self.train = train
        self.batch_size = batchSize
        self.epochs = epochs
        self.char = char
        
        
        # save the model
        if not os.path.exists(dirBin):
            os.mkdir(dirBin)
    
    def buildDNN(self, n_classes):
        """
        Build a deep neural network object tailored for the specified number of classes
        
        Parameters
        ----------   
        n_classes: int
            number of classes as output of the network
        """
        # create a model object
        model = Sequential([
            Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)),
            Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'),
            MaxPool2D(pool_size=(2,2)),
            Dropout(0.25),
            Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'),
            Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'),
            MaxPool2D(pool_size=(2,2), strides=(2,2)),
            Dropout(0.25),
            Flatten(),
            Dense(256, activation = "relu"),
            Dropout(0.5),
            Dense(n_classes, activation = "softmax")])
        
        if self.verbose == True:
            model.summary()
        
        return model
        
    
    def trainModel(self):
        """
        Train a new model with the specified attributes of the instantiated object
        """

        # load data throught the provided dataset object
        X, y = self.dataset.load()
        
        # count number of classes
        unique, _ = np.unique(y, return_counts=True)
        
        # call buildDNN method
        model = self.buildDNN(n_classes = len(unique))
        
        # preprocess data thorugh the dataset class
        X_train = self.dataset.preProcessData(X)
        y_train = self.dataset.labalEncoding(y, n_classes = len(unique))
        
        # define optimizer
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

        # compile the model
        model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics= ["accuracy"])

        # set a learning rate annealer
        learning_rate_reduction = ReduceLROnPlateau(monitor='acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
        
        # data augmentation in order to improve model performance
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images
            zoom_range = 0.1, # randomly zoom image 
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        datagen.fit(X_train)

        # fit the model
        history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=self.batch_size),
                              epochs = self.epochs, steps_per_epoch=X_train.shape[0] // self.batch_size
                              ,callbacks=[learning_rate_reduction])

        # save the model
        if self.char:
            model.save(os.path.join(self.dirBin, 'model_char.h5'))
            model.save_weights(os.path.join(self.dirBin, 'model_char_weights.h5'))
        else:
            model.save(os.path.join(self.dirBin, 'model_digit.h5'))
            model.save_weights(os.path.join(self.dirBin, 'model_digit_weights.h5'))
        logging.info('Model saved in {}'.format(self.dirBin))
        
        if self.verbose:
            self.plotHistory(history)
            
        return model
        
    
    def loadModel(self):
        """
        Try to load a model from the dirBin folder. If not found an automatic training is launched
        
        Rasises
        -------
        [WARNING] Model not found:
            if no pre-trained is found in the dirBin folder
        """

        if not self.train:
            try:
                if self.char:
                    model = load_model(os.path.join(self.dirBin, 'model_char.h5'))
                    model.load_weights(os.path.join(self.dirBin, 'model_char_weights.h5'))
                else:
                    model = load_model(os.path.join(self.dirBin, 'model_digit.h5'))
                    model.load_weights(os.path.join(self.dirBin, 'model_digit_weights.h5'))
                
                if self.verbose:
                    model.summary()
                
                return model
        
            except Exception as e:
                logging.warning('Model not found')
                logging.info('Train a new model')
                model = self.trainModel()
                return model
                
        else:
            logging.info('Train a new model')
            model = self.trainModel()
            return model
        
    def plotHistory(self, history):
        """
        Plot the loss and accuracy curves for the training dataset 
        
        Parameters
        ----------
        history: obj
            history object with all information of the training process
        """
        fig, ax = plt.subplots(2,1, figsize=(20,15))
        ax[0].plot(history.history['loss'], color='b', label="Training loss")
        legend = ax[0].legend(loc='best', shadow=True)

        ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
        legend = ax[1].legend(loc='best', shadow=True)