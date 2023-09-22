import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from config import Config, load_config
from self_driving_car_batch_generator import Generator
from utils import get_driving_styles
from utils_models import *
import argparse

np.random.seed(0)


def load_data(cfg):
    """
    Load training data_nominal and split it into training and validation set
    """
    start = time.time()

    x = None
    y = None
    path = None
    x_train = None
    y_train = None
    x_test = None
    y_test = None

    drive = get_driving_styles(cfg)
    print("Loading training set " + cfg.TRACK + str(drive))
    for drive_style in drive:
        try:
            path = os.path.join(cfg.TRAINING_DATA_DIR,
                                cfg.TRAINING_SET_DIR,
                                cfg.TRACK,
                                drive_style,
                                'driving_log.csv')
            data_df = pd.read_csv(path)
            if x is None:
                x = data_df[['center', 'left', 'right']].values
                y = data_df['steering'].values
            else:
                x = np.concatenate((x, data_df[['center', 'left', 'right']].values), axis=0)
                y = np.concatenate((y, data_df['steering'].values), axis=0)
        except FileNotFoundError:
            print("Unable to read file %s" % path)
            continue

    if x is None:
        print("No driving data_nominal were provided for training. Provide correct paths to the driving_log.csv files")
        exit()

    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=cfg.TEST_SIZE, random_state=0)
    except TypeError:
        print("Missing header to csv files")
        exit()

    duration_train = time.time() - start
    print("Loading training set completed in %s." % str(datetime.timedelta(seconds=round(duration_train))))

    print("Data set: " + str(len(x)) + " elements")
    print("Training set: " + str(len(x_train)) + " elements")
    print("Test set: " + str(len(x_test)) + " elements")
    return x_train, x_test, y_train, y_test


def train_model(model, cfg, x_train, x_test, y_train, y_test, early_stopping):
    """
    Train the self-driving car model
    """
    tracks_used = 'track'
    if len(cfg.TRACK)==3:
        tracks_used = 'all_tracks-'
    else:
        for track in cfg.TRACK:
            if '1' in track:
                tracks_used += '1-'
            elif '2' in track:
                tracks_used += '2-'
            elif '3' in track:
                tracks_used += '3-'
    
    if cfg.USE_PREDICTIVE_UNCERTAINTY:
        name = os.path.join(cfg.SDC_TRAINING_MODELS_DIR,
                            tracks_used + cfg.SDC_TRAINING_MODEL_NAME.replace('.h5', '') + '-mc' + '-{epoch:03d}.h5')
    else:
        name = os.path.join(cfg.SDC_TRAINING_MODELS_DIR,
                            tracks_used + cfg.SDC_TRAINING_MODEL_NAME.replace('.h5', '') + '-{epoch:03d}.h5')

    checkpoint = ModelCheckpoint(
        name,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='auto')

    if early_stopping:
        # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
        #                                            min_delta=.0005,
        #                                            patience=10,
        #                                            verbose=1,
        #                                            mode='auto') 
        early_stop = keras.callbacks.EarlyStopping(monitor='loss',
                                                min_delta=.0005,
                                                patience=10,
                                                mode='auto')
        callbacks=[checkpoint, early_stop]
        
        print("Early stopping active")
    else:
        callbacks=[checkpoint]
        print("Without Early stopping")   

    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=cfg.LEARNING_RATE))

    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_test, y_test = shuffle(x_test, y_test, random_state=0)

    train_generator = Generator(x_train, y_train, True, cfg)
    val_generator = Generator(x_test, y_test, False, cfg)

    history = model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=cfg.NUM_EPOCHS_SDC_MODEL,
                        callbacks=callbacks,
                        verbose=1)

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    tracks_used = 'track'
    if len(cfg.TRACK)==3:
        tracks_used = 'all_tracks-'
    else:
        for track in cfg.TRACK:
            if '1' in track:
                tracks_used += '1-'
            elif '2' in track:
                tracks_used += '2-'
            elif '3' in track:
                tracks_used += '3-'

    if cfg.USE_PREDICTIVE_UNCERTAINTY:
        name = os.path.join(cfg.SDC_TRAINING_MODELS_DIR,
                            tracks_used + cfg.SDC_TRAINING_MODEL_NAME.replace('.h5', '') + '-mc-final.h5')
    else:
        name = os.path.join(cfg.SDC_TRAINING_MODELS_DIR, tracks_used + cfg.SDC_TRAINING_MODEL_NAME.replace('.h5', '') + '-final.h5')

    # save the last model anyway (might not be the best)
    model.save(name)


def main():
    parser = argparse.ArgumentParser(description='Remote Driving - Model Training')
    parser.add_argument('-es', help='with or without early-stopping', dest='early_stopping', type=bool, default=False)
    args = parser.parse_args()
    """
    Load train/validation data_nominal set and train the model
    """
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    x_train, x_test, y_train, y_test = load_data(cfg)

    model = build_model(cfg.SDC_TRAINING_MODEL_NAME, cfg.USE_PREDICTIVE_UNCERTAINTY)

    train_model(model, cfg, x_train, x_test, y_train, y_test, args.early_stopping)


if __name__ == '__main__':
    main()
