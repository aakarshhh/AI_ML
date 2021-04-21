import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from collections import deque
from Environment import *
from Model import *
from tqdm import tqdm


if __name__ == '__main__' :
    epsilon = 1
    ed = 0.9975 
    MIN_ep = 0.0001
    MIN_rep_mem_size = 256
    episodes = 200
    STATS_EVERY = 5
    if not os.path.isdir('models'):
        os.makedirs('models')
    agent = DDQNAgent()
    env = CarlaEnv()
    time.sleep(0.1)

    scores = []
    avg_scores = []
    for episode in tqdm(range(1, episodes + 1), ascii=True, unit='episodes'):
            env.collision_hist = []
            score = 0
            step = 1
            current_state = env.reset()
            done = False
            episode_start = time.time()
            while True:
                
                if np.random.random() > epsilon:
                    #print(current_state.shape)
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    action = np.random.randint(0, 8)
                    time.sleep(1/25)
                new_state, reward, done, _ = env.step(action)
                score += reward
                agent.update_replay_memory((current_state, action, reward, new_state, done))
                current_state = new_state
                step += 1
                if done :
                    break
            for actor in env.actor_list:
                actor.destroy()
            scores.append(score)
            if not episode % STATS_EVERY or episode == 1:
                avg_scores.append(np.mean(scores[-STATS_EVERY:]))
                min_reward = min(scores[-STATS_EVERY:])
                max_reward = max(scores[-STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=avg_scores[-1], reward_min=min_reward, reward_max=max_reward, epsilon=epsilon , step = episode)
            agent.model.save('models/cnddql.model')
            print('episode:', episode, 'score %.2f' % score)
            if epsilon >= MIN_ep:
                epsilon *= ed
                epsilon = max(MIN_ep, epsilon)
    agent.terminate = True
    agent.model.save('models/cnddql.model')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(scores)
    plt.plot(avg_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
