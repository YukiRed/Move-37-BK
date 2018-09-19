import gym
from gym import wrappers
import numpy as np

env = gym.make('CartPole-v0')

bestLength = 0
episode_lengths = []

best_weight = np.zeros(4)

for i in range(10):
    new_weights = np.random.uniform(-1.0, 1.0, 4)

    length = []
    observation = env.reset()

    done = False
    counter = 0

    while not done:
        env.render()
        counter += 1
        action = 1 if np.dot(observation, new_weights) > 0 else 0
        observation, reward, done, info = env.step(action)
        if done:
            # print("Episode finished after {} timesteps".format(counter))
            break

    length.append(counter)

    average_length = float(sum(length) / len(length))

    if average_length > bestLength:
        bestLength = average_length
        best_weight = new_weights
    episode_lengths.append(average_length)

    # if i % 10 == 0:
    print('Best length is ', bestLength)


done = False
counter = 0
env = wrappers.Monitor(env, 'MovieFiles2', force=True)
observation = env.reset()

while not done:
    counter += 1
    action = 1 if np.dot(observation, best_weight) > 0 else 0
    observation, reward, done, info = env.step(action)
    print(info)
    if done:
        print('With best weight',
              "Episode finished after {} timesteps".format(counter))
        break
