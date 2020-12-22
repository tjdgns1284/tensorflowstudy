from tensorflow.keras.datasets.boston_housing import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import KFold
import numpy as np
# 0.2 의 비율로 테스트 데이터 만들고 0.8은 학습데이터 
(x_train, y_train), (x_test, y_test) = load_data(path='boston_housing.npz', test_split=0.2, seed=777 )

#data를 정규화시키는 과정 
mean = np.mean(x_train, axis = 0)
std = np.std(x_train, axis = 0)
x_train = (x_train- mean) /std
x_test = (x_test - mean) /std

#K폴드를 이용하기 
k = 3
kfold = KFold(n_splits= k, random_state= 777)

def get_model():
    model = Sequential()
    model.add(Dense(64,activation='relu',input_shape=(13,)))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss= 'mse', metrics=['mae'])

    return model

mae_list = []
for train_index,val_index in kfold.split(x_train):
    x_train_fold,x_val_fold = x_train[train_index], x_train[val_index]
    y_train_fold,y_val_fold = y_train[train_index], y_train[val_index]

    model = get_model()

    model.fit(x_train_fold, y_train_fold, epochs= 300, validation_data= (x_val_fold,y_val_fold) )

    _, test_mae = model.evaluate(x_test,y_test)
    mae_list.append(test_mae)

print(mae_list)
print(np.mean(mae_list))