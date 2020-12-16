import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets.fashion_mnist import load_data


(x_train, y_train), (x_test, y_test) = load_data()
np.random.seed(777)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

sample_size = 9 
random_idx = np.random.randint(60000,size= sample_size)

plt.figure( figsize=(5,5))
for i, idx in enumerate(random_idx):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap='gray')
    plt.xlabel(class_names[y_train[i]])

x_train = x_train/255
x_test = x_test/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train,x_val , y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=777 )
#1번모델 
first_model = Sequential()
first_model.add(Flatten(input_shape= (28,28)))
first_model.add(Dense(64,activation='relu'))
first_model.add(Dense(32,activation='relu'))
first_model.add(Dense(10,activation='softmax'))

first_model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics=['acc']
)
first_history = first_model.fit(
    x_train,
    y_train,
    epochs= 30,
    batch_size=128,
    validation_data= (x_val,y_val)
)
#2번 모델 
second_model = Sequential()
second_model.add(Flatten(input_shape= (28,28)))
second_model.add(Dense(128,activation='relu'))
second_model.add(Dense(64,activation='relu'))
second_model.add(Dense(32,activation='relu'))
second_model.add(Dense(10,activation='softmax'))

second_model.compile(
    optimizer='adam',
    loss = 'categorical_crossentropy',
    metrics=['acc']
)
second_history = second_model.fit(
    x_train,
    y_train,
    epochs= 30,
    batch_size=128,
    validation_data= (x_val,y_val)
)


def draw_loss_acc(history_1,history_2,epochs):
    his_dict_1 = history_1.history
    his_dict_2 = history_2.history
    keys = list(his_dict_1.keys())

    epochs = range(1,epochs)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params( labelcolor='w', top=False, bottom= False, left= False, right= False )

    for i in range(len(his_dict_1)):
        temp_ax = fig.add_subplot(2, 2, i+1)
        temp = keys[i%2]
        val_temp = keys[(i+2)%2 + 2 ]
        temp_history = his_dict_1 if i <2 else his_dict_2
        temp_ax.plot(epochs, temp_history[temp][1:], color= 'blue', label= 'train_' + temp)
        temp_ax.plot(epochs, temp_history[val_temp][1:], color= 'orange', label= val_temp)
        if (i == 1 or i == 3):
            start, end = temp_ax.get_ylim()
            temp_ax.yaxis.set_ticks(np.arange(np.round(start,2), end, 0.01))
        temp_ax.legend()
    ax.set_ylabel('loss',size=20)
    ax.set_xlabel('Epochs',size=20)
    plt.tight_layout()
    plt.show()

draw_loss_acc(first_history, second_history, 30)
