import tensorflow as tf
tf.random.set_seed(777)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import mse

data = np.array([[0,0],[0,1],[1,0],[1,1]])
label = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(32, input_shape = (2, ) , activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer=RMSprop(), loss= mse, metrics=['acc'])
model.fit(data,label,epochs=100)
