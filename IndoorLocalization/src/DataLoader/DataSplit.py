from numpy import int8, ndarray
from pandas import read_table
from sklearn.model_selection import train_test_split


def readDataset() -> tuple[ndarray, ndarray]:
    dataSet = read_table("data/wifi_localization.txt", header=None)
    x = dataSet.iloc[:, :-1].to_numpy(int8)
    y = dataSet.iloc[:, -1].to_numpy(int8)
    return x, y


def getTrainTestSplit(testSize: float = 0.25) -> list[ndarray]:
    """
        Returns [X_train, X_test, y_train, y_test]
    """
    x, y = readDataset()
    return train_test_split(x, y, test_size=testSize)
