import pandas as pd
import numpy  as np
import sklearn as sk
import sklearn.model_selection
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
import keras

# from https://www.kaggle.com/amarjeet007/visualize-cnn-with-keras

def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1

filepath ='./modelCNN'
model = keras.models.load_model(filepath)

layer_outputs = [layer.output for layer in model.layers]
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
X_train1 = pd.read_csv('/home/alecsoc/Desktop/mygit/EECS504_Project_F20/sign_mnist_train.csv')

X_train = X_train1.drop('label',axis=1)
X_train = np.array(X_train.values)
X_train2 = X_train / 255
X_train2 = X_train2.reshape(X_train2.shape[0],28,28,1)
activations = activation_model.predict(X_train2[10].reshape(1,28,28,1))

display_activation(activations,3,3,1)
plt.show()
display_activation(activations,3,3,2)
plt.show()
display_activation(activations,3,3,3)
plt.show()

plt.imshow(X_train2[10],cmap='gray')
plt.show()


 
