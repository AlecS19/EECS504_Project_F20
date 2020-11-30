import pandas as pd
import numpy  as np
import sklearn as sk
import sklearn.model_selection
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
import keras

filepath ='./modelCNN'
model = keras.models.load_model(filepath)
pred1 = pd.read_csv('/home/alecsoc/Desktop/mygit/EECS504_Project_F20/manual.csv')
pred = pred1.drop('label',axis=1)
pred = pred.values
pred /= 255
pred = pred.reshape(pred.shape[0],28,28,1)

prediction = model.predict(pred)
print(np.argmax(prediction, axis=1))
