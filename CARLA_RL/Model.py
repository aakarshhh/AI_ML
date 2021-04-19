import glob
import os
import sys
import random
import time
import numpy as np
import pandas as pd
import cv2
import math
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Concatenate, Conv2D, AveragePooling2D, Activation, Flatten,MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras.backend import  backend
from threading import Thread
from Environment import *
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, reward_avg, reward_min, reward_max, epsilon,step):
        with self.writer.as_default():
            
            tf.summary.scalar("reward_avg", reward_avg,step=step)
            tf.summary.scalar("reward_min", reward_min,step=step)
            tf.summary.scalar("reward_max", reward_max,step=step)
            tf.summary.scalar("epsilon", epsilon,step=step)
        self.writer.flush()
        
        
class DQNAgent:
    def __init__(self):
        REPLAY_MEMORY_SIZE = 2048
        MINIBATCH_SIZE = 256
        PREDICTION_BATCH_SIZE = 1
        TRAINING_BATCH_SIZE = 16
        UPDATE_TARGET_EVERY = 5
        MIN_REWARD = -1
        
        DISCOUNT = 0.9875
        MIN_REPLAY_MEMORY_SIZE = 1024
        first_time = True 

        if first_time:
            self.model = self.create_model()
        else :
            self.model = load_model('models/cndrl.model')
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/cndrl-{int(time.time())}")
        self.target_update_counter = 0
        self.alpha = 0.001
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        IM_WIDTH = 640
        IM_HEIGHT = 480
        model = Sequential()
        model.add(Conv2D(64, (5, 5), input_shape=(480, 640,3), padding='same',data_format='channels_last'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Dropout(.2))
        model.add(Conv2D(64,(5, 5)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Dropout(.25))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))
        model.add(Flatten())
        model.add(Dense(16))
        model.add(tf.keras.layers.Dropout(.2))
        model.add(Dense(8))
        model.compile(loss="mse", optimizer=Adam(), metrics=["accuracy"])
        return model
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    def train(self):
        oq = pd.read_csv('q_value.csv')
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE :
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch])/255
        
        current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        
        future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)
        x = []
        y = []
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):             
            max_future_q = np.max(future_qs_list[index])
            oq[f'{action}'] = (1-self.alpha)*oq[f'{action}']+ self.alpha*(reward + DISCOUNT * max_future_q)
            current_qs = current_qs_list[index]
            current_qs[action] = oq[f'{action}']
            x.append(current_state)
            y.append(current_qs)

        log_this_step = False
        oq.to_csv('q_value.csv',index = False)
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        self.model.fit(np.array(x)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0 , shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state))/255)

    def train_in_loop(self):
        self.training_initialized = True
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)
        

