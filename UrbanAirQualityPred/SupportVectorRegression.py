import matplotlib.pyplot as plt
from numpy import int16, concatenate, array
from pandas import read_csv
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import Nystroem


class SupportVectorRegression(object):

    def __init__(self) -> None:
        self.data = read_csv("./UrbanAirQualityPred/Beijingair.csv",
                             header=None).to_numpy(int16)
        step = 1
        # rows, cols = self.data.shape
        rows, cols = 30000, 10
        X = [self.data[i:i + step, :].flatten() for i in range(0, rows - step)]
        self.predictors = [
            SGDRegressor("epsilon_insensitive") for _ in range(cols)
        ]
        self.featureMap = Nystroem(n_components=300)
        X = self.featureMap.fit_transform(X)
        for j in range(cols):
            y = [self.data[i, j] for i in range(step, rows)]
            self.predictors[j].fit(X, y)
        self.X_test = self.data[30000 - step:30000, :]
        self.fX_test = self.featureMap.transform([self.X_test.flatten()])
        y_pred = [None for _ in range(1000)]
        for i in range(0, 1000):
            y_pred[i] = concatenate([
                self.predictors[k].predict(self.fX_test).astype(int16)
                for k in range(cols)
            ])
            self.X_test = concatenate([self.X_test, [y_pred[i]]], axis=0)[-step:, :]
            print(self.X_test)
            self.fX_test = self.featureMap.transform([self.X_test.flatten()])
        y_pred = array(y_pred, dtype=int16)
        for i in range(cols):
            y_true = self.data[30000:31000, i]
            plt.subplot(2, 5, i + 1)
            plt.plot(y_pred[:, i], label="pred")
            plt.plot(y_true, linestyle='--', label="true")
            plt.legend()

        plt.show()


SupportVectorRegression()
