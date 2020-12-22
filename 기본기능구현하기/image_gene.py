# 데이터 준비해오기 
import pandas as pd


DATA_PATH = './csv_data/nocolorinfo'

train_df = pd.read_csv(DATA_PATH+'/train.csv')
val_df = pd.read_csv(DATA_PATH+'/val.csv')
test_df = pd.read_csv(DATA_PATH+'/test.csv')


#제너레이터정의 모델구현하기 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten


train_datagen =  ImageDataGenerator(rescale=1./255)
val_datagen =  ImageDataGenerator(rescale=1./255)

def get_steps(num_samples, batch_size):
    if (num_samples%batch_size) >0 : 
        return (num_samples//batch_size) +1 
    else: 
        return num_samples//batch_size

model = Sequential()
model.add(Flatten(input_shape= (112,112,3)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(11, activation='sigmoid'))

model.compile( optimizer='adam', loss= 'binary_crossentropy', metrics= ['acc']) 

batch_size = 32 
class_col = [
    'black','white','brown','blue','red','green',
    'dress','shirt','pants','shorts','shoes'
] 

train_generator= train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='',
    x_col='image',
    y_col=class_col,
    target_size=(112,112),
    color_mode='rgb',
    class_mode='raw',
    batch_size=batch_size,
    seed=42
)

val_generator= train_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory='',
    x_col='image',
    y_col=class_col,
    target_size=(112,112),
    color_mode='rgb',
    class_mode='raw',
    batch_size=batch_size,
    shuffle=True
)
print(train_generator)
model.fit(
    train_generator,
    steps_per_epoch=get_steps(len(train_df),batch_size),
    validation_data= val_generator,
    validation_steps= get_steps(len(val_df),batch_size),
    epochs= 10

)

test_datagen = ImageDataGenerator(rescale = 1./255)


test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df, 
    directory='',
    x_col = 'image',
    y_col = None,
    target_size = (112, 112),
    color_mode='rgb',
    class_mode=None,
    batch_size=batch_size,
    shuffle = False
)

import matplotlib.pyplot as plt
#이거왜이래
import cv2
preds = model.predict(test_generator,
                     steps = get_steps(len(test_df), batch_size),
                     verbose = 1)
do_preds = preds[:8]

for i, pred in enumerate(do_preds):
    plt.subplot(2,4,i+1)
    prob = zip(class_col,list(pred))
    prob = sorted(list(prob), key= lambda z: z[1], reverse=True)[:2]

    image = cv2.imread(test_df['image'][i])
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.title(f'{prob[0][0]}: {round(prob[0][1] * 100, 2)}% \n {prob[1][0]}: {round(prob[1][1] * 100, 2)}%')
    
plt.tight_layout()