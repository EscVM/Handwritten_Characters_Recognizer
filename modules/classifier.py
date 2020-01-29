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


import numpy as np
import matplotlib.pyplot
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import logging


# configure the level of logging to INFO
logging.basicConfig(level=logging.INFO)


class DigitClassificator(object):
    """
    DigitClassificator loads a model object and performs the needed operations to properly 
    prepare it for inference. Then two methods are available to predict characters from plain 
    sheets or boxes with a specific dimension (Exam project).
    
    ...
    
    Attributes
    ----------
    modelObj: obj
        instance of the Model class
    verbose: bool
        if True print some useful information
 
    Methods
    -------
    loadPreImage(img_path, resize=False, h=950)
        Load an image from a specified path and applies conversion if needed
    predict(img_path, char=False)
        Detect characters on a plain sheet of paper through the trained model
    predictBoxes(img_path, char=False)
        Detect characters on a sheet of paper with boxes of a specific dimension (Exam project)
    """
    def __init__(self, ModelObj, verbose=False):
        """
        Parameters
        ----------
        
        modelObj: obj
            instance of the Model class
        verbose: bool
            if True print some useful information
        """
        
        self.model = ModelObj
        self.classifier = self.model.loadModel()
        # ASCII code for labels translation
        self.labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
                       75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 100, 101, 102,
                       103, 104, 110, 113, 114, 116]
        
    def loadPreImage(self, img_path, resize=False, h=950):
        """
        Load an image from a specified path and applies conversion if needed
        
        Parameters
        ----------
        img_path: str
            formatted string with the path of the image to process
        resize: bool
            if true input, loaded image is resized to a certain dimension keeping constant the aspect ratio
        h: int
            image dimension of the height
        
        Rasises
        -------
        [ERROR] if OPENCV is not able to load the specified image on memory
        """
        
        try:
            # read image
            img = cv2.imread(img_path)
            # resize if needed
            if resize:
                # keep aspect ratio
                r = h / img.shape[1]
                dim = (h, int(img.shape[0] * r))
                img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
            return img
        
        except Exception as e:
            # print opencv error
            logging.error(e)
            exit(1)
    
    def predict(self, img_path, char=False):
        """
        Detect characters on a plain sheet of paper through the trained model
        
        Parameters
        ----------
        img_path: str
            formatted string with the path of the image to process
        char: bool
            if True ASCII table is used to code predictions
            
        Rasises
        -------
        [ERROR] if predictions go wrong
        """
        
        # load a new image
        img = self.loadPreImage(img_path, resize=True)
        
        # Convert to grayscale and apply Gaussian filtering
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # Threshold the image
        ret, img_th = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the image
        ctrs, hier = cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get rectangles contains each contour
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]

        # For each rectangular region apply the trained classifier
        for rect in rects:
            # Make the rectangular region around the found character
            leng = int(rect[3] * 1.1)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = img_th[pt1:pt1+leng, pt2:pt2+leng]
            # try to predict the selected box
            try:
                # Resize the image
                roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                # add padding
                roi = cv2.dilate(roi, (4, 4))
                # normalize roi
                roi = roi / 255
                # predict the image
                y_pred = self.classifier.predict(roi[None,...,None])
                
                # Draw rectangles
                cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 3) 
                if char:
                    # add label using ASCII table
                    cv2.putText(img, str(chr(self.labels[np.argmax(y_pred)])), (rect[0] + rect[2]//2, rect[1] -                                                                 rect[2]//6), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 0), 3)
                else:
                    # add label without using ASCII table (only numbers)
                    cv2.putText(img, str(np.argmax(y_pred)), (rect[0] + rect[2]//2, rect[1] -                                                                                rect[2]//6), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 0), 3)
                
            except Exception as e:
                logging.error(e)
                
        # plot the final picture      
        plt.figure(figsize=(20,20))
        plt.imshow(img)
        plt.show()
        
    def predictBoxes(self, img_path, char=False):
        """
        Detect characters on a sheet of paper with boxes of a specific dimension (Exam project)
        
        Parameters
        ----------
        img_path: str
            formatted string with the path of the image to process
        char: bool
            if True ASCII table is used to code predictions
            
        Rasises
        -------
        [ERROR] if predictions go wrong
        
        """
        # load a new image
        img = self.loadPreImage(img_path, resize=False)
        
        # Convert to grayscale and apply Gaussian filtering
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
        
        # Threshold the image
        ret,thresh = cv2.threshold(img_gray, 127, 255, 1)

        # Find contours in the image
        contours,h = cv2.findContours(thresh, 1, 2)

        # For each rectangular region apply the trained classifier
        for cnt in contours:
            # find approx curves
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            # if curve has four points is a square
            if len(approx)==4:
                # find rectangle
                (x, y, w, h) = cv2.boundingRect(approx) # top left width height
                # find the right ratio (can be tuned for the specific project)
                area = w * h
                aspect_ratio = w / float(h)
                if (area >= 0.001 * img.shape[0] * img.shape[1]) and aspect_ratio <= 1.7:
                    roi = thresh.copy()
                    roi = roi[y+7:y+h-7,x+7:x+w-7]
                    # try to predict the selected box
                    try:
                        # Resize the image
                        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                        # normalize the roi
                        roi = roi / 255.
                        # predict with the model
                        y_pred = self.classifier.predict(roi[None,...,None])
                        # Draw the rectangles
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        if char:
                            # add label using ASCII table
                            cv2.putText(img, str(chr(self.labels[np.argmax(y_pred)])), (x + w//2, y -                                                                                  w//6), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 0), 3)
                        else:
                            # add label without using ASCII table (only numbers)
                            cv2.putText(img, str(np.argmax(y_pred)), (x + w//2, y - w//6), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255,                                                             0), 3)
                    except Exception as e:
                        logging.error(e)
        
                    
        # plot the final picture 
        plt.figure(figsize=(20,20))
        plt.imshow(img)
        plt.show()