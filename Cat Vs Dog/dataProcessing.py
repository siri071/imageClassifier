#libraries
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import pickle

#give directory 
Directory = r'' #write your directory here
categories = ['cats','dogs']

#appending all the images and label into data variable
img_size = 100
data = []
for category in categories:
    folder = os.path.join(Directory,category)
    label = categories.index(category) #label of class
    for img in os.listdir(folder):
        img_path = os.path.join(folder,img)
        #print(img_path)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr,(img_size,img_size)) #change img size to make the sizes same
        data.append([img_arr,label])
    
#print(len(data))
#shuffle the data to make model more accurate
random.shuffle(data)
X,Y = [],[]
for features,labels in data:
    X.append(features)       
    Y.append(labels)

X = np.array(X) #features in form of array(3D)
Y = np.array(Y) #label --> cat or dog

pickle.dump(X,open('X.pkl','wb'))
pickle.dump(Y,open('Y.pkl','wb'))







    