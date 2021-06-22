from sklearn.preprocessing import MinMaxScaler

class Preprocess:

    def scaling(self,data):
        self.data = data
        scaler = MinMaxScaler()
        self.scaled = scaler.fit_transform(self.data.values.reshape(-1,1))
        return scaler #returning object of minmax scaler because we can inverse it while using in testing(main.py)

    def create_batches(self ,b):
        x = []
        y = []
        

        for i in range(b,len(self.scaled)):
            x.append(self.scaled[i-b:i,0].reshape(-1,1))
            y.append(self.scaled[i-1,0])
        return x,y
if __name__=='__main__':
    pass
