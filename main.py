import pandas
from stocks_prediction.cleaning import Preprocess
from stocks_prediction.model import model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def forecast():
    n=5

    batch = x_train[-1].reshape(1,5,1)
    forecast =[]

    for i in range(n):
        y_pred_new = model.predict(batch)
        forecast.append(y_pred_new)
        batch = np.append(batch,y_pred_new)[1:].reshape(1,5,1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1,1)).flatten()
    return forecast


if __name__=='__main__':
    data = pd.read_csv('TSLA.csv')
    preprocess =  Preprocess()
    scaler  = preprocess.scaling(data['close'])
    # preparing data in batches
    batch_size = 5
    x,y = preprocess.create_batches(batch_size)

    
    x_train,x_test, y_train,y_test = train_test_split(x,y ,test_size = 0.2 ,shuffle=False)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model = model(batch_size)
    model.fit(x_train,y_train ,epochs = 10 ,batch_size = 5)

    print('Training Done...')

    forecast = forecast()
    y_test = scaler.inverse_transform(y_test.reshape(-1,1))
    print('Next 5 timestamp predicted: ', forecast)
    print('Actual Values',y_test[:5])