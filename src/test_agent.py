import gymnasium as gym
# import gym
import panda_gym
import numpy as np
import imageio
import time
from torch import nn

from vecenv_actor import ActorNeuralNetwork


lr = 1e-4
# env_id = 'PandaReach-v3'
env_id = 'PandaPickAndPlace-v3'
# env_id = 'PandaPush-v3'
# env_id = 'PandaPush-v2'
state_shape = 0
env = gym.make(env_id, render=True)
env.metadata['render_fps'] = 29
buffer_type = "HER"
if (isinstance(env.observation_space, gym.spaces.Dict)):
    for _, _obs_shape in env.observation_space.items():
        state_shape = state_shape + _obs_shape.shape[-1]
else:
    state_shape = env.observation_space.shape[-1]
nn_dims = [256,256,256]
# folder = 'tmp/networks/'
folder = 'tmp/checkpoints/'
file_name = '_Actor.pth'
# file_name = '1.0_DDPG_Iter_1_Actor.pth'
# folder = 'tmp/best_models/'
# file_name = 'DDPG_Iter_4_Actor.pth'

has_bias = True
global_actor = ActorNeuralNetwork(env=env, input_dims=state_shape, n_actions=env.action_space.shape[-1], lr=lr, nn_dims=nn_dims, name="Actor", 
                                  chkpt_dir=folder, add_bias=has_bias)

if has_bias:
    global_actor.load_model(path=folder, file_name=file_name)
    for layer in global_actor.children():
        if isinstance(layer, nn.Linear):
            print(f"Layer Weights: {layer.state_dict()['weight']}")
            print(f"Layer {layer} Biases: {layer.state_dict()['bias']}")
        
num_episodes = 10
'''Panda Gym Environment'''
env.metadata['render_fps'] = 15
save_gif = False

if save_gif:
    '''Save simulation as GIF'''
    images = []
    obs, info = env.reset()
    while np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'], axis=-1) <= 0.05:
            obs, info = env.reset()
    img = env.render(mode="rgb_array")
    for i in range(350):
        images.append(img)
        action = global_actor.forward(obs).cpu().detach().data.numpy()
        obs, _, ter, trun ,_ = env.step(action*1.0)
        img = env.render(mode="rgb_array")
        if ter or trun:
            [images.append(img) for _ in range(10)]
            obs, info = env.reset()
            while np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'], axis=-1) <= 0.05:
                obs, info = env.reset()

    env.close()
    date_now = time.strftime("%Y%m%d%H%M")
    # imageio.mimsave("tmp/GIF/" + date_now + "_" + env_id + ".gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=20)
    imageio.mimsave("tmp/GIF/" + date_now + "_" + env_id + ".gif", [np.array(img) for i, img in enumerate(images)], fps=15)
else:
    '''Render Simulation'''
    for episode in range(10):
        state, info = env.reset()
        # state = env.reset()
        while np.linalg.norm(state['achieved_goal'] - state['desired_goal'], axis=-1) <= 0.05: # or state['desired_goal'][2] <=0.04:
            state, info = env.reset()
        done, acc_reward = False, 0
        desired_tag = "achieved_goal"
        while not done:
            env.render(mode='human')
            '''Panda Gym Environment'''
            action = global_actor.forward(state).cpu().detach().data.numpy()
            state, reward, terminated, truncated, info = env.step(action)
            
            acc_reward += reward
            done = terminated | truncated
        
        '''Panda Gym Environment'''
        print("Episode {:4d} out of {:d} -> Reward: {:.4f}.\tInfo: {}\tZ desired: {}".format(episode+1, num_episodes, acc_reward, info, state['desired_goal'][2]))
    env.close()
