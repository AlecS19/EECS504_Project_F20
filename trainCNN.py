import pandas as pd
import numpy  as np
import sklearn as sk
import sklearn.model_selection
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
import keras

######################################################
#Inspiration from the following medium article
# https://medium.com/@romainvrh/sign-language-classifier-using-cnn-e5c2fb99ef51
######################################################

normal = False
filepath = './modelCNN'


## Import data
if normal:
    train = pd.read_csv('/home/alecsoc/Desktop/eecs504/project/sign_mnist_train/sign_mnist_train.csv')
    test = pd.read_csv('/home/alecsoc/Desktop/eecs504/project/sign_mnist_test/sign_mnist_test.csv')
else:
    train = pd.read_csv('/home/alecsoc/Desktop/mygit/EECS504_Project_F20/mnist_rotated_train.csv')
    test = pd.read_csv('/home/alecsoc/Desktop/mygit/EECS504_Project_F20/mnist_rotated_test.csv')
    

## Dataframe to nparray
labels = train['label'].values

images = train.drop('label',axis=1)
images = images.values

images = np.array([np.reshape(i,(28,28))for i in images])

images = np.array([i.flatten() for i in images])

## Pre-process

# One-hot coding
label_binrizer = sk.preprocessing.LabelBinarizer()
labels = label_binrizer.fit_transform(labels)

p = 0.2 #proportion of test data for validation
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(images,labels, test_size=p,random_state=101)

# Convert from 0-255 to 0-1
x_train = x_train/255
x_test = x_test/255

# bias term 
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

# hyperparamters
batch_size = 128
num_classes = 24
epochs = 10

## Model Set-up
model = keras.Sequential()

#CNN
model.add(keras.layers.Conv2D(16,kernel_size=(3,3), activation='relu',input_shape=(28,28,1) ))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

#CNN
model.add(keras.layers.Conv2D(32,kernel_size=(3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

#CNN
model.add(keras.layers.Conv2D(64,kernel_size=(3,3), activation='relu' ))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

#Fully-connected    
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.10))

#softmax
model.add(keras.layers.Dense(num_classes, activation='softmax'))

#cost function
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(), metrics=['accuracy'])  

history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size, validation_split=0.1)

#saving model
keras.models.save_model(model,filepath)

## Plotting
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()