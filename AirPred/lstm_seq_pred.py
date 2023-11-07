from os import path
from typing import Tuple

import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from numpy import array, ndarray, reshape
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


class LSTMSeqPred:
    """
    Implementation of Urban Air Quality Pred through RNN with LSTM
    """

    def __init__(self) -> None:
        self.input_sample_len = 24
        X, Y = LSTMSeqPred.init_data(input_sample_len=self.input_sample_len)
        self.data = LSTMSeqPred.split_data(X, Y)
        self.model = LSTMSeqPred.init_model(
            input_sample_len=self.input_sample_len)
        self.optim = Adam(learning_rate=0.0002)
        self.model.compile(optimizer=self.optim, loss="mse", metrics=['mae'])
        self.train()
        self.multi_pred()

    @staticmethod
    def init_data(file_path: str = "./AirPred/Beijingair.csv",
                  input_sample_len: int = 12) -> Tuple[ndarray]:
        """
        Load original dataset from file `Beijingair.csv` and initialize new dataset for prediction.
        """
        data = scale(reshape(read_csv(file_path).values, (31875, 10)))
        input_seq, output_seq = [], []
        for i in range(31875 - input_sample_len):
            x = data[i:i + input_sample_len]
            y = data[i + input_sample_len]
            input_seq.append(x)
            output_seq.append(y)
        return array(input_seq), array(output_seq)

    @staticmethod
    def split_data(input_seq_mat: ndarray,
                   output_seq_mat: ndarray) -> Tuple[Tuple[ndarray]]:
        """
        split dataset into train, test and val subsets at ratios of `(0.64, 0.16, 0.2)`
        """
        x_train, x_test, y_train, y_test = train_test_split(input_seq_mat,
                                                            output_seq_mat,
                                                            test_size=0.2,
                                                            shuffle=False)
        x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                          y_train,
                                                          test_size=0.2,
                                                          shuffle=False)
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    @staticmethod
    def init_model(input_sample_len: int = 12) -> Sequential:
        """
        Initialize RNN-LSTM model architecture
        """
        model = Sequential(
            [LSTM(64, input_shape=(input_sample_len, 10)),
             Dense(10)])
        return model

    def train(self, force_train: bool = False):
        """
        A method which trains the model and plots afterwise.
        """
        if not force_train and path.exists("./AirPred/lstm_model.keras"):
            self.model.load_weights("./AirPred/lstm_model.keras")
            return
        history = self.model.fit(x=self.data[0][0],
                                 y=self.data[0][1],
                                 batch_size=64,
                                 epochs=35,
                                 validation_data=self.data[1],
                                 shuffle=False)
        self.model.save("./AirPred/lstm_model.keras")
        _, test_mae = self.model.evaluate(*self.data[2])
        plt.plot(history.history['mae'], label="mae")
        plt.plot(history.history['val_mae'], label="val_mae")
        plt.plot([test_mae] * 32, c='r', linestyle="--", label="test_mae")
        plt.xlim((-0.5, 29.5))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("./AirPred/plots/loss.png")

    def multi_pred(self):
        """
        multi-seq prediction and plotting
        """
        headers = [
            "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "Temp.", "Pressure",
            "Dew Temp.", "Wind Speed"
        ]
        y_pred_test = self.model.predict(self.data[2][0])
        for i, header in enumerate(headers):
            plt.figure(figsize=(20, 10))
            plt.plot(y_pred_test[:500, i], c='b', label="Pred")
            plt.plot(self.data[2][1][:500, i],
                     c='black',
                     linestyle='dotted',
                     label="Truth")
            plt.title(header)
            plt.xlabel("h")
            plt.ylabel("Value")
            plt.xlim((-0.5, 500.5))
            plt.legend()
            plt.savefig(f"./AirPred/plots/{i}_{header}.png")


LSTMSeqPred()
