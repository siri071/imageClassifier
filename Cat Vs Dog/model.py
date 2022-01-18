import time
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard

#Tensorboard is used for data visualization
#Better and efficient than matplotlib and seaborn.
NAME = f'cat-vs-dog-prediction-{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs\\{NAME}\\')


X = pickle.load(open('X.pkl','rb'))
Y = pickle.load(open('Y.pkl','rb'))
#print(X,Y)

#feature scaling to reduce the pixels to make our model faster
X = X/255

#print(X.shape) --> (100(h),100(w),3(color channel))

#CNN sequential model
model = Sequential()
model.add(Conv2D(64,(3,3),activation='relu',input_shape = X.shape[1:]))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

#Flatten --> convert the feature array from m x n to m x 1.
#Dense --> Neural network consists of neurons (eg. 64,128,etc)
model.add(Flatten())
model.add(Dense(128,input_shape = X.shape[1:],activation ='relu'))
model.add(Dense(128,activation ='relu'))
model.add(Dense(2,activation='softmax')) # gives the value between 0 to 1 and 2 represents the no. of outputs

model.compile(optimizer='adam',
loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

model.fit(X,Y,epochs=5,validation_split = 0.1,batch_size=32,callbacks=[tensorboard])

