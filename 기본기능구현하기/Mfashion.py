import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split


tf.random.set_seed(777)

(x_train, y_train), (x_test, y_test) = load_data(path='mnist.npz')

x_train,x_val , y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=777 )

num_x_train = x_train.shape[0]
num_x_val = x_val.shape[0]
num_x_test = x_test.shape[0]

x_train = (x_train.reshape((num_x_train,28*28)))/255
x_val = (x_val.reshape((num_x_val,28*28)))/255
x_test = (x_test.reshape((num_x_test,28*28)))/255

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(64,activation='relu', input_shape= (784,)))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss= 'categorical_crossentropy' , metrics=['acc'])
history = model.fit (x_train,y_train , epochs= 30 , batch_size = 128, validation_data =(x_val,y_val))

# model.evaluate(x_test,y_test)
result = model.predict(x_test)