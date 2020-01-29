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
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from tensorflow.keras.datasets import mnist
from emnist import extract_test_samples
import numpy as np
import matplotlib.pyplot
from keras.utils.np_utils import to_categorical
import os
import matplotlib.pyplot as plt
import math
import logging

# configure the level of logging to INFO
logging.basicConfig(level=logging.INFO)



class Dataset(object):
    """
    A class used to Download training data using the emnist package. Setting properly the 
    different attributes is possible to download different kinds of dataset ( for example containing only digits).
    It downloads only ones the dataset and then it caches (~500 MB) a zip folder in the default .cache folder.
    A couple of methods give the possibility to explore the loaded dataset
    
    ...
    
    Attributes
    ----------
    mainDir: str
        path of the main folder of the project
    dirData: str
        path of the dataset directory where downloaded data are stored
    onlyDigits: bool
        if True a dataset of only digits is loaded
    kerasDB (only if onlyDigits is True): bool
        if True it downloads the original MNIST dataset
    verbose: bool
        if True print some useful information
 
    Methods
    -------
    load()
        Downloads the dataset following the specifification of the instantiated instance
    preProcessData(X)
        Normalize input dasta (X) and add a fourth dimension
    labalEncoding(labels, n_classes)
        Encode the label vector in a one-hot-encoder array
    plotDigits(images_batch, labels, img_n)
        Plot a certain number of training examples in a grid 
    printInfo(y)
        Print how many examples per class are in the dataset
    """
    def __init__(self,mainDir, dirData = 'dataset', onlyDigits = False, kerasDB = False, verbose = False):
        """
        Parameters
        ----------
        mainDir: str
            path of the main folder of the project
        dirData: str
            path of the dataset directory where downloaded data are stored
        onlyDigits: bool
            if True a dataset of only digits is loaded
        kerasDB (only if onlyDigits is True): bool
            if True it downloads the original MNIST dataset
        verbose: bool
            if True print some useful information
        """
 
        self.dirData = os.path.join(mainDir, dirData)
        self.verbose = verbose
        self.onlyDigits = onlyDigits
        self.kerasDB = kerasDB
        
        # if not present make a new dataset folder
        if not os.path.exists(dirData):
            logging.info('Making a new Dataset Directory')
            os.mkdir(dirData)
        
    def load(self):
        """
        Downloads the dataset following the specifification of the instantiated instance
        """
        # donwload proper dataset
        if self.onlyDigits:
            logging.info('Downloading digits from EMNIST repository')
            X, y = extract_test_samples('digits')
            if self.kerasDB:
                logging.info('Downloading the Original MNIST dataset from TensorFlow Datasets')
                (X_train, y_train), (X_test, y_test) = mnist.load_data()
                # concatenate train and test together
                # we want to perform the training with the highest number of training instances
                X = np.concatenate((X_train, X_test))
                y = np.concatenate((y_train, y_test))
        else:
            logging.info('Downloading letters and digits from EMNIST repository')
            X, y = extract_test_samples('balanced')
        # save downloded data in the dataset folder as numpy array
        np.save(os.path.join(self.dirData, 'data.npy'), X)
        np.save(os.path.join(self.dirData, 'label.npy'), y)
        
        if self.verbose:
            self.printInfo(y)
        
        return X, y
        
    def preProcessData(self, X):
        """
        Normalize input dasta (X) and add a fourth dimension
        
        Parameters
        ---------   
        X: numpy array (int8)
            number of classes as output of the network
        """
        # normalize the data (max value is 255)
        X = X / 255.0
        # add a fourth dimension
        X = X.reshape(-1, 28, 28, 1)
        
        return X
    
    def labalEncoding(self, labels, n_classes):
        """
        Encode the label vector in a one-hot-encoder array
        
        Parameters
        ---------   
        labels: numpy array (int8)
            label array
        num_classes: int
            number of classes 
        """
        return to_categorical(labels, num_classes = n_classes)
                
    def plotDigits(self, images_batch, labels, img_n):
        """
        Plot a certain number of training examples in a grid 
        
        Parameters
        ---------  
        images_batch: numpy array (int8)
            tensor with training data
        labels: numpy array (int*)
            tensor with training labels
        img_n:
            number of images to plot
        """
        # number of columns
        max_c = 5

        # find the number of columns and rows
        if img_n <= max_c:
            r = 1
            c = img_n
        else:
            r = math.ceil(img_n/max_c)
            c = max_c

        # plot images in a grid form with matplolib
        fig, axes = plt.subplots(r, c, figsize=(15,15))
        axes = axes.flatten()
        for img_batch, label_batch, ax in zip(images_batch, labels, axes):
            ax.imshow(img_batch)
            ax.grid()
            ax.set_title('Class: {}'.format(label_batch))
        plt.tight_layout()
        plt.show()
    
    def printInfo(self, y):
        """
        Print how many examples per class are in the dataset
        
        Parameters
        ---------  
        y: numpy array (int8)
            tensor with training labels
        """
        # count how many unique instances
        unique, counts = np.unique(y, return_counts=True)
        # print info
        for i, j in zip(unique, counts):
            print('Class {0}: {1} elements'.format(i, j))
        print('-' * 9)