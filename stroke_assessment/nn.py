import os, glob
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

import stroke_assessment
from stroke_assessment.preprocess import PreprocessData


def network(train_X: pd.DataFrame, train_y: pd.DataFrame, val_X: pd.DataFrame, val_y: pd.DataFrame):
    print(train_X)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(train_X.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        loss='binary_crossentropy',
        metrics=['acc'],
        optimizer='adam'
    )

    history = model.fit(x=train_X, y=train_y, validation_data=(val_X, val_y),
              epochs=10)


if __name__ == '__main__':
    train_data = pd.read_csv(stroke_assessment.TRAIN_DATA, index_col='id')
    test_data = pd.read_csv(stroke_assessment.TEST_DATA, index_col='id')

    train_X = PreprocessData(train_data).output_data_
    train_y = train_data[['stroke']]
    train_X = train_X.drop('stroke', axis=1)

    # test_X = PreprocessData(test_data).output_data_
    #
    # train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2)
    #
    # # network(train_X, train_y, val_X, val_y)
    # train_X.to_csv(os.path.join(stroke_assessment.DATA_DIR, 'data.csv'))
    # print(train_X.isnull().any())
