"""
This file implements a feedforward neural network using tf.keras. Data are
preprocessed using :py:mod:`preprocess` and then used in training. A series of flags
are available to change the behaviour of this script and are described in the comments.

In breif however, this script can :
    1) Train the model and monitor overfitting with a validation set
    2) Evaluate the model on validation data
    3) Since test labels are not available, we cannot evaluate on test data
    4) Bootstrap model training assess stability of model performance


"""
import os, glob
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import stroke_assessment
from stroke_assessment.preprocess import PreprocessData, UnderSample


# todo use a confusion matrix
# todo use a ROC curve
# todo bootstrap the results with random sample for calculating confidence level


def prepare_data(test_data, train_data) -> tuple:
    """
    Simple wrapper around other functions to prepare data for input to nn.

    Args:
        test_data (pd.DataFrame):
        train_data (pd.DataFrame):

    Returns: train_X, train_y, val_X, val_y, test_X

    """
    test_X = PreprocessData(test_data).output_data_
    train_X = PreprocessData(train_data).output_data_

    # train_X, val_X = PreprocessData.under_sample_with_val_split(train_X, val_split)
    train_X, val_X = UnderSample.under_sample_age_strat_with_val_split(train_X)

    train_y = train_X[['stroke']]
    train_X = train_X.drop('stroke', axis=1)
    val_y = val_X[['stroke']]
    val_X = val_X.drop('stroke', axis=1)

    return train_X, train_y, val_X, val_y, test_X  # no test_y is given


def network(train_X: pd.DataFrame, train_y: pd.DataFrame, val_X: pd.DataFrame, val_y: pd.DataFrame,
            dropout_rate=0.2, epochs=100, use_early_stopping=False):
    """
    Train a neural network
    Args:
        train_X:
        train_y:
        val_X:
        val_y:
        dropout_rate:
        epochs:
        use_early_stopping:

    Returns:

    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(train_X.shape[1],)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        loss='binary_crossentropy',
        metrics=['acc'],
        optimizer='adam'
    )

    history = model.fit(
        x=train_X, y=train_y, validation_data=(val_X, val_y),
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=25, min_delta=5,
            )
        ] if use_early_stopping else []
    )

    return model, pd.DataFrame(history.history)


def plot_history(history, savefig=False):
    """
    plot fitting history
    Args:
        history:
        savefig:

    Returns:

    """
    history['epoch'] = range(history.shape[0])
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    sns.lineplot(x='epoch', y='loss', data=history, label='loss')
    sns.lineplot(x='epoch', y='val_loss', data=history, label='val_loss')
    plt.subplot(1, 2, 2)
    sns.lineplot(x='epoch', y='acc', data=history, label='acc')
    sns.lineplot(x='epoch', y='val_acc', data=history, label='val_acc')

    sns.despine(fig=fig, top=True, right=True)
    if savefig:
        plt.savefig(stroke_assessment.HISTORY_PLOT_FILE, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    # some flags to modify this scripts behaviour

    # When false, load model from previous run (will error if not exist).
    #  Otherwise, train a new model and save to MODEL_FILE
    TRAIN_NEW_MODEL = True

    # boolean indicator for whether to plot history
    PLOT_HISTORY = True

    # flag to again evaluate the model on validation data
    EVALUATE = True

    # resample many times
    BOOTSTRAP_NETWORK = True

    # number of times to bootstrap
    NBOOT = 100

    # some hyperparameters
    val_split = 0.2
    epochs = 1000
    dropout_rate = 0.5
    use_early_stopping = True

    # read in data
    train_data = pd.read_csv(stroke_assessment.TRAIN_DATA, index_col='id')
    test_data = pd.read_csv(stroke_assessment.TEST_DATA, index_col='id')

    # split data into train and val data
    train_X, train_y, val_X, val_y, test_X = prepare_data(test_data, train_data)  # no test_y available

    if TRAIN_NEW_MODEL:
        model, history = network(
            train_X, train_y, val_X, val_y, epochs=epochs,
            dropout_rate=dropout_rate,
            use_early_stopping=use_early_stopping
        )
        tf.keras.models.save_model(model, stroke_assessment.MODEL_FILE)
        history.to_csv(stroke_assessment.HISTORY_FILE, index=False)

    else:
        model = tf.keras.models.load_model(stroke_assessment.MODEL_FILE)
        history = pd.read_csv(stroke_assessment.HISTORY_FILE)

    if PLOT_HISTORY:
        plot_history(history, savefig=True)

    if EVALUATE:
        eval = model.evaluate(val_X, val_y)
        print(f'The model has {eval[1] * 100:.2f}% accuracy')

    if BOOTSTRAP_NETWORK:
        models = []
        histories = []
        for i in range(NBOOT):
            train_X, train_y, val_X, val_y, test_X = prepare_data(test_data, train_data)  # no test_y available
            model, history = network(
                train_X, train_y, val_X, val_y, epochs=epochs,
                dropout_rate=dropout_rate,
                use_early_stopping=use_early_stopping
            )
            models.append(model)
            histories.append(history.iloc[[-1]])
        hist = pd.concat(histories).reset_index()
        # save to file
        hist.to_csv(stroke_assessment.BOOTSTRAP_HISTORY_FILE)

        print(hist)
        print(hist.describe())

    # print(pred)
