import pandas as pd
import numpy as np


class PairedRegression:
    def __init__(self, data, column1, column2):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data должна быть объектом типа DataFrame")

        data = data.dropna()

        X = data[column1].values
        Y = data[column2].values

        X = X[~np.isnan(X.astype(float))]
        Y = Y[~np.isnan(Y.astype(float))]

        if not (len(X) == len(Y)):
            raise ValueError("X и Y разного размера")

        # Вычисляем суммы
        sumX = sum(X)
        sumY = sum(Y)
        sumXY = sum(X * Y)
        sumXX = sum(X * X)
        n = len(X)

        self.b1 = (sumXY - (sumX * sumY) / n) / (sumXX - sumX * sumX / n)
        self.b0 = (sumY - self.b1 * sumX) / n

    def predict(self, X):
        X = X[~np.isnan(X.astype(float))]
        predictions = self.b0 + self.b1 * X
        return predictions