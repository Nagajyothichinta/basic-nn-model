# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network with multiple hidden layers and multiple nodes in each hidden layer is known as a deep learning system or a deep neural network. Here the basic neural network model has been created with one input layer, one hidden layer and one output layer.The number of neurons(UNITS) in each layer varies the 1st input layer has 16 units and hidden layer has 8 units and output layer has one unit.

In this basic NN Model, we have used "relu" activation function in input and hidden layer, relu(RECTIFIED LINEAR UNIT) Activation function is a piece-wise linear function that will output the input directly if it is positive and zero if it is negative.


## Neural Network Model

![image](https://github.com/Nagajyothichinta/basic-nn-model/assets/94191344/25fdd5df-5e66-48a2-8617-af345862a500)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('ex1').sheet1
data = worksheet.get_all_values()
dataset1=pd.DataFrame(data[1:],columns=data[0])
dataset1=dataset1.astype({'input':'float'})
dataset1=dataset1.astype({'output':'float'})
dataset1.head()
X = dataset1[['input']].values
y = dataset1[['output']].values
X
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
ai_brain=Sequential([
    Dense(8,activation='relu'),
    Dense(10,activation='relu'),
    Dense(1)
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(X_train1,y_train,epochs=200)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
```
## Dataset Information


<img width="206" alt="image" src="https://github.com/Nagajyothichinta/basic-nn-model/assets/94191344/7d583125-408c-4747-afcb-b2ed31ac29d0">


## OUTPUT

Include your plot here

<img width="416" alt="image" src="https://github.com/Nagajyothichinta/basic-nn-model/assets/94191344/d6bf7fe9-a9d6-4dd5-a90f-da1b560d9cc1">


### Test Data Root Mean Squared Error

<img width="410" alt="image" src="https://github.com/Nagajyothichinta/basic-nn-model/assets/94191344/fdff8d7f-65d5-4648-963f-ab3a9e9b80e2">


### New Sample Data Prediction

<img width="344" alt="image" src="https://github.com/Nagajyothichinta/basic-nn-model/assets/94191344/38cbb6a6-7224-41bd-8863-799479da9e35">


## RESULT

A Basic neural network regression model for the given dataset is developed successfully.
