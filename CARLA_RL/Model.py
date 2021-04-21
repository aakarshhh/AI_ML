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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
    def set_model(self, model):
        pass
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)
    def on_batch_end(self, batch, logs=None):
        pass
    def on_train_end(self, _):
        pass

    def update_stats(self, reward_avg, reward_min, reward_max, epsilon,step):
        with self.writer.as_default():
            tf.summary.scalar("reward_avg", reward_avg,step=step)
            tf.summary.scalar("reward_min", reward_min,step=step)
            tf.summary.scalar("reward_max", reward_max,step=step)
            tf.summary.scalar("epsilon", epsilon,step=step)
            self.writer.flush()
                
class DDQNAgent:
    def __init__(self):
        rep_mem_size = 1024
        mb_size = 128        
        D = 0.9875
        MIN_rep_mem_size = 256
        first_time = False 
        if first_time:
            self.model = self.create_model()
        else :
            self.model = load_model('models/cnddql.model')
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=rep_mem_size)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/cnddql-{int(time.time())}")
        self.target_update_counter = 0
        self.alpha = 0.001
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        IM_WIDTH = 640
        IM_HEIGHT = 480
        model = Sequential()
        model.add(Conv2D(64, (5, 5), input_shape=(480 , 640 , 4), padding='same'))
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
        model.add(Dense(8, activation="linear" ))
        model.compile(loss="mse", optimizer=Adam(), metrics=["accuracy"])
        return model
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    def train(self):
        oq = pd.read_csv('q_value.csv')
        if len(self.replay_memory) < MIN_rep_mem_size :
            return
        minibatch = random.sample(self.replay_memory, mb_size)
        current_states = np.array([transition[0] for transition in minibatch])/255
        
        current_qs_list = self.model.predict(current_states, 1)
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        
        future_qs_list = self.target_model.predict(new_current_states, 1)
        x = []
        y = []
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            
            max_future_q = np.max(future_qs_list[index])
            oq[f'{action}'] = (1-self.alpha)*oq[f'{action}']+ self.alpha*(reward + D*max_future_q)
            current_qs = current_qs_list[index]
            current_qs[action] = oq[f'{action}']
            x.append(current_state)
            y.append(current_qs)
            print(oq)
        log_this_step = False
        oq.to_csv('q_value.csv',index = False)
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        self.model.fit(np.array(x), np.array(y), batch_size=4, verbose=0 , shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > 5:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, state.shape[0], state.shape[1], state.shape[2])/255)

    def train_in_loop(self):
        self.training_initialized = True
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)
        #self.tensorboard --logdir= logs/
        

