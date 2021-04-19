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
from threading import Thread
from Environment import *
from Model import *
from tqdm import tqdm

if __name__ == '__main__' :
    REPLAY_MEMORY_SIZE = 2048
    MIN_REPLAY_MEMORY_SIZE = 1024
    MIN_REWARD = -1
    epsilon = 1
    EPSILON_DECAY = 0.9975 
    MIN_EPSILON = 0.0001
    EPISODES = 100
    AGGREGATE_STATS_EVERY = 10
    FPS = 20
    # For stats
    ep_rewards = [-1]
    if not os.path.isdir('models'):
        os.makedirs('models')

    agent = DQNAgent()
    env = CarEnv()
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.05)
    scores = []
    avg_scores = []
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
            env.collision_hist = []
            score = 0
            step = 1
            current_state = env.reset()
            done = False
            episode_start = time.time()
            while True:
                if np.random.random() > epsilon:
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    action = np.random.randint(0, 8)
                    time.sleep(1/FPS)
                new_state, reward, done, _ = env.step(action)
                score += reward
                # Every step we update replay memory
                agent.update_replay_memory((current_state, action, reward, new_state, done))
                current_state = new_state
                step += 1
                if done :
                    break
            for actor in env.actor_list:
                actor.destroy()

            scores.append(score)

            if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                avg_scores.append(np.mean(scores[-AGGREGATE_STATS_EVERY:]))
                min_reward = min(scores[-AGGREGATE_STATS_EVERY:])
                max_reward = max(scores[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=avg_scores[-1], reward_min=min_reward, reward_max=max_reward, epsilon=epsilon , step = episode)
            agent.model.save('models/cndrl.model')

            print('episode: ', episode, 'score %.2f' % score)
            if epsilon >= MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)
    agent.terminate = True
    trainer_thread.join()
    agent.model.save('models/cndrl.model')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(scores)
    plt.plot(avg_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
