from osim.env import L2M2019Env
import numpy as np

env = L2M2019Env(difficulty=3, visualize=False)
env.change_model(model='3D', difficulty=3)

observation = env.reset(project=True, obs_as_dict=True)
total_reward = 0.0
num_episodes = 2

v_target = observation['v_tgt_field'] # <class 'numpy.ndarray'> (2, 11, 11)

for x in range(num_episodes):
    episode_over = False
    while not episode_over:
        obs, current_reward, episode_over, info = env.step(env.action_space.sample(), project=True)
        # print(env.v_tgt_field)
        total_reward += current_reward
        # print(total_reward)

