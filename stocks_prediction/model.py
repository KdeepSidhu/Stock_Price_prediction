from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def model(b):
    model = Sequential()
    model.add(LSTM(units=50,input_shape=(b,1)))
    model.add(Dense(150,activation ='relu'))
    model.add(Dense(1))
    model.compile(optimizer = 'adam',loss = 'MSE')
    model.summary()

    return model 