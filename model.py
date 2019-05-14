

import numpy as np
import csv

from keras.models import Sequential
from keras.layers import Cropping2D,Convolution2D, Flatten, Lambda
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam



import nv_generator as gn
from sklearn.model_selection import train_test_split

np.random.seed(0)

csv_file='./driving_log_nom.csv'

samples = []

with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        samples.append(row)
        
csv_file='./driving_log_nom2.csv'

with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        samples.append(row)
        
csv_file='./driving_log_rec.csv'

with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        samples.append(row)

csv_file='./driving_log_rev.csv'

with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        samples.append(row)
             

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print(len(train_samples))
print(len(validation_samples))

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2, 2), name='Conv1'))
model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2, 2), name='Conv2'))
model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2, 2), name='Conv3'))
model.add(Convolution2D(64, 3, 3, activation='elu', name='Conv4'))
model.add(Convolution2D(64, 3, 3, activation='elu', name='Conv5'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation='elu', name='FC1'))
model.add(Dense(50, activation='elu', name='FC2'))
model.add(Dense(10, activation='elu', name='FC3'))
model.add(Dense(1))
model.summary()

#------------------------------- model.load_weights('./nv/weights_lz_nv0_v1.h5')

train_generator = gn.generator(train_samples)
validation_generator = gn.generator(validation_samples)

compile
opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss='mse', metrics=[])

history = model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), verbose=2,nb_epoch=10)
 
print('Model Works')
 
model.save('./model_SX.h5')
model.save_weights('./weights_SX.h5')

from keras import backend as K
K.get_session().close()
K.clear_session()

import tensorflow as tf
tf.keras.backend.clear_session()