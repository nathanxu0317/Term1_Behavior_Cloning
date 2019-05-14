"""
Created on Sun Dec 3 2017

@author: Trucker
"""


import cv2
import numpy as np
#import matplotlib.image as mpimg
import sklearn.utils as sk
from random import shuffle

def random_flip(image, angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """

    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
        angle = -angle

    return image, angle


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """

    if np.random.rand()<0.3:
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:,:,2] =  hsv[:,:,2] * ratio
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return image

def random_RGB(image):
    """
    Randomly adjust RGB of the image.
    """

    if np.random.rand()<0.3:
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        i=np.random.randint(3)
        image[:,:,i] =  image[:,:,i] * ratio

        
    return image

def generator(samples, batch_size=4):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        
        shuffle(samples)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                steering_center = float(batch_sample[3])
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                img_center = batch_sample[0]
                img_left = batch_sample[1].strip()
                img_right = batch_sample[2].strip()
                
                img_temp=[img_center, img_left, img_right]
                steering_temp=[steering_center,steering_left,steering_right]
                i=np.random.randint(3)
                
                image = np.asarray(cv2.imread(img_temp[i]))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                angle = steering_temp[i]
                
                image = random_brightness(image)
                image = random_RGB(image)
                image,angle = random_flip(image, angle)
                
                #===============================================================
                # image = preprocess(image)
                #===============================================================
                
                images.append(image)
                angles.append(angle)  

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sk.shuffle(X_train, y_train)
            
            
            
            
            
            
