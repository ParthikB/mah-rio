# This file is used to generate random training_data.
import retro
import cv2
import numpy as np
from handlers import *
import os
import time


env = retro.make('SuperMarioBros-Nes')

info(env)

total_reward = 0
training_data = []
score_log = []

for episode in range(EPISODES):
    env.reset()
    ep_score = 0
    ep_start = time.time()
    print("Episode no. : ", episode+1)
    while True:
        # env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        state = cv2.cvtColor(np.array(state), cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state, (120, 112))

        ep_score += reward

        if reward > 0:
            training_data.append([state, action])

        if done:
            print("Game over | Episode score : ", ep_score)
            print(f"Runtime : {round(time.time() - ep_start, 2)} seconds.")
            print("---------------------------------------")
            break

        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    score_log.append(ep_score)

os.chdir("C:/Users/ParthikB/PycharmProjects/retro/training_data")
np.save(DATA_NAME, training_data)

env.close()

