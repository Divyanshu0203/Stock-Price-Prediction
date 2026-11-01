"""Two LSTM models sharing one backbone:
  - regressor : predicts the next day's (scaled) log return
  - classifier: predicts the probability that the next day closes up
"""
import keras
from keras import Input, Sequential
from keras.layers import LSTM, Dense, Dropout


def _backbone(window, n_features):
    return [
        Input(shape=(window, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
    ]


def build_regressor(window, n_features):
    model = Sequential(_backbone(window, n_features) + [Dense(16, activation="relu"), Dense(1)])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mean_squared_error")
    return model


def build_classifier(window, n_features):
    model = Sequential(
        _backbone(window, n_features) + [Dense(16, activation="relu"), Dense(1, activation="sigmoid")]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model