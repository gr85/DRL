from typing import Dict, List, OrderedDict, Union
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import json
import time
import os
from tqdm import tqdm
import shutil
from math import dist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from copy import deepcopy

from noise import NormalNoise, OUNoise
from vecenv_buffer import experienceReplayBuffer, HERBuffer
from vecenv_herBuffer import HerReplayBuffer
from vecenv_actor import ActorNeuralNetwork
from vecenv_critic import CriticNeuralNetwork
from utils import StateVecToList

import gymnasium as gym
# import gym
import panda_gym


class TD3_RandProp(object):
    def __init__(self, env, buffer, noise:str="OUNoise", theta:float=0.15, sigma:float=0.2, dt:float=0.1, folder:str='tmp/networks',
                 gamma:float=0.99, mem_size:int=int(1e6), burn_in_tsteps:int=int(25e3), batch_size:int=100, lr:float=0.001, polyak:float=0.99, upd_freq:int=2,
                 random_prop:float=0.3, nn_dims:List=[256,256,256], env_id='PandaPickAndPlace-v3', n_envs:int=1, iteration:int=0, epsilon:float=1., bias:bool=True):
        self.folder = folder
        self.gamma = gamma
        self.env = env
        self.update_freq = upd_freq
        self.buffer_type = buffer_type
        self.burn_in_tsteps = burn_in_tsteps
        self.random_prob = random_prop
        self.validate_timesteps = 0
        self.env_id = env_id
        self.ts_offset = 0
        # In order to force exploration use epsilon-greedy approach
        self.epsilon = epsilon
        self.epsion_decay = 0.999
        self.best_succ_rate = 0.0
        self.iteration = iteration + 1
        
        '''Panda-Gym V3'''
        self.single_env = gym.make(self.env_id)
        '''Panda-Gym V2'''
        # self.single_env = gym.make(self.env_id, control_type='ee')
        
        if noise == "OUNoise":
            self.noise = OUNoise(theta=theta, mu=np.zeros((n_envs, self.env.single_action_space.shape[-1])), sigma=sigma, dt=dt)
        else:
            self.noise = NormalNoise(sigma=sigma, size=self.env.single_action_space.shape[-1], n_envs=n_envs)
            
        state_shape = 0        
        if (isinstance(env.single_observation_space, gym.spaces.Dict)):
            for key, _obs_shape in env.single_observation_space.items():
                # if key == 'observation' or key == 'desired_goal':
                #     state_shape = state_shape + _obs_shape.shape[-1]          
                state_shape = state_shape + _obs_shape.shape[-1]  
        else:
            state_shape = env.single_observation_space.shape[-1]
        
        self.buffer = buffer
        '''stable-baselines3'''
        self.buffer.reset()
        self.batch_size = batch_size
        self.a_lb = env.single_action_space.low
        self.a_ub = env.single_action_space.high
        self.polyak = polyak
        
        self.update_loss = [] 
        self.update_critic_loss = []
        self.loss_evolution = [] # La pèrdua durant l'entrenament
        self.critic_loss_evolution = [] # La pèrdua durant l'entrenament
        self.training_rewards = [] # Les recompenses obtingudes a cada pas de l'entrenament
        self.mean_training_rewards = [] # Les recompenses mitjanes cada 100 episodis
        self.success_rate = [] # Average success rate on validaton model
        self.success_rate_ts = [] # Timestep which validation were performed
        
        '''Neural Networks for Panda Gym Environments'''
        self.actor = ActorNeuralNetwork(env=self.single_env, input_dims=state_shape, n_actions=env.single_action_space.shape[-1], lr=lr, nn_dims=nn_dims, name="Actor", 
                                        chkpt_dir=self.folder+'/networks', add_bias=bias)
        self.critic_1 = CriticNeuralNetwork(env=self.single_env, input_dims=state_shape+env.single_action_space.shape[-1], name="Critic_1", n_outs=1, lr=lr, nn_dims=nn_dims, 
                                            chkpt_dir=self.folder+'/networks', add_bias=bias)
        self.critic_2 = CriticNeuralNetwork(env=self.single_env, input_dims=state_shape+env.single_action_space.shape[-1], name="Critic_2", n_outs=1, lr=lr, nn_dims=nn_dims,
                                            chkpt_dir=self.folder+'/networks', add_bias=bias)
        self.target_actor = ActorNeuralNetwork(env=self.single_env, input_dims=state_shape, n_actions=env.single_action_space.shape[-1], lr=lr, nn_dims=nn_dims, name="Target_Actor", 
                                               chkpt_dir=self.folder+'/networks', add_bias=bias)
        self.target_critic_1 = CriticNeuralNetwork(env=self.single_env, input_dims=state_shape+env.single_action_space.shape[-1], name="Target_Critic_1", n_outs=1, lr=lr, nn_dims=nn_dims,
                                                   chkpt_dir=self.folder+'/networks', add_bias=bias)
        self.target_critic_2 = CriticNeuralNetwork(env=self.single_env, input_dims=state_shape+env.single_action_space.shape[-1], name="Target_Critic_2", n_outs=1, lr=lr, nn_dims=nn_dims,
                                                   chkpt_dir=self.folder+'/networks', add_bias=bias)
        
        self.update_network_parameters(polyak=0) # polyak=0 to copy exactly the same params at the first time
        
    def get_action(self, state, evaluation:bool=False, random:bool=False):       
        if random:
            return np.array((self.env.action_space.sample() + 
                            self.noise.get_sample()).clip(self.a_lb, self.a_ub), dtype=float) # acció random
                
        if not evaluation:   
            if np.random.random() < self.epsilon: # self.random_prob:
                actions = np.array((self.env.action_space.sample() + 
                            self.noise.get_sample()).clip(self.a_lb, self.a_ub), dtype=float) # acció random  
            else:
                actions = (self.actor.forward(state).cpu().detach().data.numpy() + 
                            self.noise.get_sample()).clip(self.a_lb, self.a_ub)                     
        else:
            actions = [self.actor.forward(s).clip(self.a_lb, self.a_ub) for s in state]      
               
        return actions
      
    def train(self, max_tsteps=1000, update_steps=-1, tsteps_checkpoint=1000, load_from_checkpoint:bool=False, validate_timesteps:int=80, validate_eps:int=10):
        '''
        if update_steps <= 0 -> Update the networks at the end of each episode
        '''
        eps_rewards, steps = 0, 0
        self.episodes = 0
        self.validate_timesteps = validate_timesteps
        
        _load_from_checkpoint = load_from_checkpoint
        
        # Omplim el buffer amb N experiències aleatòries ()
        # if not _load_from_checkpoint:
        print("Filling replay buffer...")
        '''Panda-Gym V3'''
        state, info = self.env.reset()
        '''Panda-Gym V2'''
        # state = self.env.reset()
        # while np.linalg.norm(state['achieved_goal'] - state['desired_goal'], axis=-1) <= 0.05:
        #     state, info = self.env.reset()
        self.noise.reset()
        done = False
        timesteps = 0
        while timesteps < self.burn_in_tsteps:
            timesteps += 1
            
            action = self.get_action(state, random=True)
            
            '''Panda-Gym V3'''
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated | truncated
            '''Panda-Gym V2'''
            # next_state, reward, done, info = self.env.step(action)
            # any_success = np.array([i['is_success'] for i in info], dtype=bool)
            # done = done | any_success
           
            # self.buffer.append(state, action, reward, next_state, done)
            '''stable-baselines3'''
            self.buffer.add(obs=state, next_obs=next_state, action=action, reward=reward, done=done, infos=info, is_virtual=False)
            state = next_state.copy()                            
        
        print('Starting training...')
        if _load_from_checkpoint:
            self.load_checkpoint(folder=self.folder)
            max_tsteps = self.episodes
        
        done = False
        eps_rewards = 0
        eps_completed = 0
        '''Panda-Gym V3'''
        state, info = self.env.reset()
        '''Panda-Gym V2'''
        # state = self.env.reset()
        # while np.linalg.norm(state['achieved_goal'] - state['desired_goal'], axis=-1) < 0.07:
        #     state, info = self.env.reset()
        self.noise.reset() 
        for timesteps in tqdm(range(max_tsteps)):           
            steps += 1
            action = self.get_action(StateVecToList(state))  
            # action = self.get_action(state)  
            
            '''Panda-Gym V2'''
            # next_state, reward, done, info = self.env.step(action)
            # any_done = any(done)==True
            '''Validation done based on episodes completed'''
            # if any_done or any([i['is_success'] for i in info])==1.0:
            #     eps_completed += 1
            #     if eps_completed % validate_timesteps == 0:
            #         print('\t... Performing Validation of Model ...')
            #         self.success_rate.append(self.validate_model(n_eps=validate_eps))
            #         self.success_rate_ts.append(episode)
            #         print(f'\tSuccess Rate: {self.success_rate[-1]}')   
            
            '''Panda-Gym V3'''            
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated | truncated
            
            # Only update epsilon if the first environment has ended the episode
            if done[0]:
                # Actualitzar epsilon segons la velocitat de decaïment fixada
                self.epsilon = max(self.epsilon * self.epsion_decay, self.random_prob)
                
            '''Validation done based on episodes completed'''
            # if any(done)==True:
            #     eps_completed += 1
            #     if eps_completed % validate_timesteps == 0:
            #         print('\t... Performing Validation of Model ...')
            #         self.success_rate.append(self.validate_model(n_eps=validate_eps))
            #         self.success_rate_ts.append(timesteps)
            #         print(f'\tSuccess Rate: {self.success_rate[-1]}')
            '''Validation done based on timesteps achieved'''
            if (timesteps+1) % validate_timesteps == 0:
                print('\t... Performing Validation of Model ...')
                success_rate = self.validate_model(n_eps=validate_eps)
                if success_rate >= self.best_succ_rate:
                    self.best_succ_rate = success_rate
                    self.actor.save_checkpoint(folder='tmp/best_models', filename_header='TD3_Iter_'+str(self.iteration))
                self.success_rate.append(success_rate)
                self.success_rate_ts.append(timesteps)
                print(f'\tSuccess Rate: {self.success_rate[-1]}')
            
            eps_rewards += reward
            
            # self.buffer.append(state, action, reward, next_state, done)
            '''stable-baselines3'''
            self.buffer.add(state, next_state, action, reward, done, info)
            self.update_networks(steps)
            state = next_state.copy()
            
            # self.loss_evolution.append(np.mean(self.update_loss))
            # self.critic_loss_evolution.append(np.mean(self.update_critic_loss))
            # self.update_loss = []
            # self.update_critic_loss = []
                       
            if timesteps % tsteps_checkpoint == 0 and timesteps > 1:
                self.episodes = max_tsteps - timesteps
                self.save_checkpoint()
                # print(f'Episode {episode} -> Success Rate: {self.success_rate[-1]*100.0} %')
        print('\t... Performing Final Validation of Model ...')
        if success_rate >= self.best_succ_rate:
            self.best_succ_rate = success_rate
            self.actor.save_checkpoint(folder='tmp/best_models', filename_header='TD3_Iter_'+str(self.iteration)+'_FinalModel')
        self.success_rate.append(success_rate)
        self.success_rate_ts.append(timesteps)
        print(f'\tSuccess Rate of Final Validation: {self.success_rate[-1]}')
                                
    def update_networks(self, timesteps, batch=None):
        # state, action, reward, next_state, done = self.buffer.sample_batch(batch_size=self.batch_size)
        '''stable-baselines3'''
        samples = buffer.sample(batch_size=self.batch_size, env=self.env)
        # print(f'observations: {samples[0]["observation"].shape}')
        # print(f'actions: {samples[1]}')
        # print(f'next_observations: {samples[2]}')
        # print(f'dones: {samples[3]}')
        # print(f'rewards: {samples[4]}')
        state = samples[0]
        action = samples[1]
        next_state = samples[2]
        done = samples[3]      
        reward = samples[4]
        
        # Separem les variables de l'experiència i les convertim a tensors
        if (isinstance(self.env.observation_space, gym.spaces.Dict)):
            # states = state
            # next_states = next_state
            states = StateVecToList(state)
            next_states = StateVecToList(next_state)
        else:
            states = np.array(state) #torch.tensor(np.array(state), dtype=torch.float).to(self.actor.device)
            next_states = np.array(next_state) #torch.tensor(np.array(next_state), dtype=torch.float).to(self.actor.device)
            
        # actions = torch.tensor(np.array(action), dtype=torch.float).to(self.actor.device)
        # rewards = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        # dones = torch.tensor(done, dtype=torch.float).to(self.actor.device)
        '''stable-baselines3'''
        actions = action
        rewards = reward
        dones = done

        # next_states = StateVecToList(next_states)
        with torch.no_grad():     
            noise_1 = torch.randn_like(actions).to(device=self.target_actor.device)
            noise_2 = torch.FloatTensor(np.array(self.noise.sigma)).to(device=self.target_actor.device)
            noise = noise_1 * noise_2
            noise = noise.clip(torch.FloatTensor(-self.noise.ll[0]).to(device=self.target_actor.device), 
                   torch.FloatTensor(self.noise.hl[0]).to(device=self.target_actor.device)).to(device=self.target_actor.device)
            next_action = (
                self.target_actor.forward(next_states) + noise
                ).clip(torch.FloatTensor(self.a_lb).to(device=self.target_actor.device), torch.FloatTensor(self.a_ub).to(device=self.target_actor.device))
       
            # next_action = (
            #     self.target_actor.forward(next_states).to(device=self.actor.device) + self.noise.get_sample().to(device=self.actor.device)
            #     ).clip(torch.tensor(self.a_lb).to(device=self.actor.device), torch.tensor(self.a_ub).to(device=self.actor.device))
            # Compute the target Q value
            target_Q1 = self.target_critic_1.forward(next_states, next_action)
            target_Q2 = self.target_critic_2.forward(next_states, next_action)
            target_Q = rewards + (1.0 - dones) * self.gamma * torch.min(target_Q1, target_Q2)
           
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic_1.forward(states, actions), self.critic_2.forward(states, actions)
        # Compute critic loss
        # critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        critic_loss_1 = F.mse_loss(current_Q1, target_Q)
        critic_loss_2 = F.mse_loss(current_Q2, target_Q)
        # crtici_loss = min(critic_loss_1, critic_loss_2)
        
		# Optimize the critic 1
        self.critic_1.optimizer.zero_grad()
        critic_loss_1.backward()
        self.critic_1.optimizer.step()
        
        # Optimize the critic 2
        # self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        # crtici_loss.backward()
        # critic_loss_1.backward()
        critic_loss_2.backward()
        # self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        if (timesteps % self.update_freq) == 0:
            # Update actor and target networks delayed        
            actor_loss = -self.critic_1.forward(states, self.actor.forward(states)).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            
            self.update_loss.append(actor_loss.cpu().detach().numpy())
            self.update_critic_loss.append((critic_loss_1 + critic_loss_2).cpu().detach().numpy())
            self.loss_evolution.append(actor_loss.cpu().detach().numpy())
            self.critic_loss_evolution.append((critic_loss_1 + critic_loss_2).cpu().detach().numpy())
            
            self.update_network_parameters()    
        
    def update_network_parameters(self, polyak=None):
        if polyak is None:
            polyak = self.polyak
            
        actor_params = self.actor.named_parameters()
        critic1_params = self.critic_1.named_parameters()
        critic2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic1_params = self.target_critic_1.named_parameters()
        target_critic2_params = self.target_critic_2.named_parameters()
        
        actor_state_dict = dict(actor_params)
        critic1_state_dict = dict(critic1_params)
        critic2_state_dict = dict(critic2_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic1_state_dict = dict(target_critic1_params)
        target_critic2_state_dict = dict(target_critic2_params)
        
        for name in critic1_state_dict:
            critic1_state_dict[name] = polyak*target_critic1_state_dict[name].clone() + (1-polyak)*critic1_state_dict[name].clone()
            
        for name in critic2_state_dict:
            critic2_state_dict[name] = polyak*target_critic2_state_dict[name].clone() + (1-polyak)*critic2_state_dict[name].clone()
            
        for name in actor_state_dict:
            actor_state_dict[name] = polyak*target_actor_state_dict[name].clone() + (1-polyak)*actor_state_dict[name].clone()
            
        self.target_critic_1.load_state_dict(critic1_state_dict)
        self.target_critic_2.load_state_dict(critic2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        
    def validate_model(self, n_eps=100):
        success = 0
        self.actor.eval()
        for _ in range(n_eps):
            '''Panda-Gym V3'''
            state, info = self.single_env.reset()
            while np.linalg.norm(state['achieved_goal'] - state['desired_goal'], axis=-1) <= 0.05:
                state, info = self.single_env.reset()
            done = False
            while not done:
                action = self.actor.forward(state).cpu().detach().data.numpy()
                state, _, terminated, truncated, _ = self.single_env.step(action)                
                if terminated:
                    success += 1
                    done = True
                elif truncated:
                    done = True 
            '''Panda-Gym V2'''
            # state = self.single_env.reset()
            # while np.linalg.norm(state['achieved_goal'] - state['desired_goal'], axis=-1) <= 0.05:
            #     state = self.single_env.reset()
            # done = False            
            # while not done:
            #     action = self.actor.forward(state).cpu().detach().data.numpy()
            #     state, _, done, info = self.single_env.step(action)                
            #     if info['is_success']==1.0:
            #         success += 1
            #         done = True
                
        self.actor.train()
        return 1.0*success/n_eps
       
    def save_models(self):
        self.actor.save_model(env_id='TD3_'+self.env_id)
        self.target_actor.save_model(env_id='TD3_'+self.env_id)
        self.critic_1.save_model(env_id='TD3_'+self.env_id)
        self.critic_2.save_model(env_id='TD3_'+self.env_id)
        self.target_critic_1.save_model(env_id='TD3_'+self.env_id)
        self.target_critic_2.save_model(env_id='TD3_'+self.env_id)
        
    def load_models(self, path='tmp\\networks', file_name=None):
        self.actor.load_model(path=path, file_name=file_name[0])
        self.target_actor.load_model(path=path, file_name=file_name[1])
        self.critic_1.load_model(path=path, file_name=file_name[2])
        self.critic_2.load_model(path=path, file_name=file_name[3])
        self.target_critic_1.load_model(path=path, file_name=file_name[4])
        self.target_critic_2.load_model(path=path, file_name=file_name[5])
        
    def save_checkpoint(self):  
        # fetch all files
        for file_name in os.listdir(self.folder+'/checkpoints/'):
            # construct full file path
            source = self.folder + '/checkpoints/' + file_name
            destination = self.folder + '/checkpoints_aux/' + file_name
            # copy only files
            if os.path.isfile(source):
                shutil.copy(source, destination)
                print('copied', file_name)            
        if not os.path.isdir(self.folder+'/checkpoints'):
            print('... Creating checkpoint folder ...')
            os.mkdir(self.folder+'/checkpoints')
        
        print('... Saving chekcpoint ....')
        self.actor.save_checkpoint(folder=self.folder+'/checkpoints', filename_header='')
        self.target_actor.save_checkpoint(folder=self.folder+'/checkpoints', filename_header='')
        self.critic_1.save_checkpoint(folder=self.folder+'/checkpoints', filename_header='')
        self.critic_2.save_checkpoint(folder=self.folder+'/checkpoints', filename_header='')
        self.target_critic_1.save_checkpoint(folder=self.folder+'/checkpoints', filename_header='')
        self.target_critic_2.save_checkpoint(folder=self.folder+'/checkpoints', filename_header='')
        # self.buffer.save_checkpoint(folder=self.folder+'/checkpoints')
        
        print('... Saving training information ...')
        dict_params = {"mean_loss": np.array(self.loss_evolution).tolist(), 
                       "train_reward": self.training_rewards, 
                       "mean_train_reward": self.mean_training_rewards,
                       "mean_critic_loss": np.array(self.critic_loss_evolution).tolist(),
                       "success_rate": self.success_rate,
                       "success_rate_ts": self.success_rate_ts,
                       "actor_filename": '_' + self.actor.file_name,
                       "target_actor_filename": '_' + self.target_actor.file_name,
                       "critic1_filename": '_' + self.critic_1.file_name,
                       "critic2_filename": '_' + self.critic_2.file_name,
                       "target_critic1_filename": '_' + self.target_critic_1.file_name,
                       "target_critic2_filename": '_' + self.target_critic_2.file_name,
                       "episodes": self.episodes 
                       }
        
        with open(self.folder+'/checkpoints/td3_train_info.json', 'w') as wf:
            json.dump(dict_params, wf)
    
    def load_checkpoint(self, folder='tmp'):
        print('... Loading values from checkpoint ...')
        with open(folder+'/checkpoints/td3_train_info.json') as rf:
            dict_res = json.load(rf)
        
        print('mean loss ...')
        self.loss_evolution = dict_res['mean_loss']
        print('train reward ...')
        self.training_rewards = dict_res['train_reward'] 
        print('mean train reward ...')
        self.mean_training_rewards = dict_res['mean_train_reward'] 
        print('mean critic loss ...')
        self.critic_loss_evolution = dict_res['mean_critic_loss'] 
        print('success rate ...')
        self.success_rate = dict_res['success_rate'] 
        print('success rate timesteps ...')
        self.success_rate_ts = dict_res['success_rate_ts'] 
        print('episodes ...')
        self.episodes = dict_res['episodes']
        
        print('Neural Nets ...')
        self.actor.load_checkpoint(dict_res['actor_filename'], folder=folder)
        self.target_actor.load_checkpoint(dict_res['target_actor_filename'], folder=folder)
        self.critic_1.load_checkpoint(dict_res['critic1_filename'], folder=folder)
        self.critic_2.load_checkpoint(dict_res['critic2_filename'], folder=folder)
        self.target_critic_1.load_checkpoint(dict_res['target_critic1_filename'], folder=folder)
        self.target_critic_2.load_checkpoint(dict_res['target_critic2_filename'], folder=folder)
        
        # print('Buffer ...')
        # self.buffer.load_checkpoint(folder=folder)
        
    def save_results(self, folder='tmp'):
        dict_params = {"mean_loss": [str(el) for el in self.loss_evolution],
                       "train_reward": [str(el) for el in self.training_rewards],
                       "mean_train_reward": [str(el) for el in self.mean_training_rewards],
                       "mean_critic_loss": [str(el) for el in self.critic_loss_evolution],
                       "success_rate": [str(el) for el in self.success_rate],
                       "success_rate_ts": self.success_rate_ts
                       }
        
        date_now = time.strftime("%Y%m%d%H%M")
        with open(folder+'/train_res/'+date_now+'_'+self.env_id+'_td3_train_vals.json', 'w') as wf:
            json.dump(dict_params, wf)        
            
    def load_results(self, file):
        with open(file) as rf:
            dict_res = json.load(rf)
        return dict_res
    
    def plot_results(self, file='tmp/train_res/td3_train_vals.json'):
        data = self.load_results(file)
        
        # Grafico la posició final, l'evolució mitjana de les pèrdues i l'evolució d'epsilon
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1)

        ax1 = plt.subplot(gs[0,:])
        ax1.plot([float(d) for d in data['mean_loss']], label='Mean Actor Loss', color=(0., 0., 1., 1.))
        ax1.plot([float(d) for d in data['mean_critic_loss']], label='Mean Critic Loss', color=(1., 0., 0., 0.6))        
        ax1.set_title('Mean Loss per Episode')
        ax1.set_xlabel('Time steps')
        ax1.set_ylabel('Loss')
        # ax1.set_xscale('log')
        ax1.legend(loc="upper left")

        ax2 = plt.subplot(gs[1,:])
        ax2.plot([float(d) for d in data['train_reward']], label='Rewards', color=(0., 0., 1., 0.3))
        ax2.plot([float(d) for d in data['mean_train_reward']], label='Mean Rewards', color=(0., 0., 1., 1.))
        ax2.set_title('Rewards per Episode')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Rewards')
        # ax2.set_xscale('log')
        ax2.legend(loc="upper left")
        
        fig.add_subplot(ax1)
        fig.add_subplot(ax2)
        
        plt.show()
        
        plt.plot([float(d) for d in data['mean_loss']], label='Mean Actor Loss', color=(0., 0., 1., 1.))
        plt.plot([float(d) for d in data['mean_critic_loss']], label='Mean Critic Loss', color=(1., 0., 0., 0.6))
        plt.title('Mean Loss per Episode')
        plt.xlabel('Time steps')
        plt.ylabel('Loss')
        plt.legend(loc="upper left")
        # plt.xscale('log')
        plt.show()

        plt.plot([float(d) for d in data['train_reward']], label='Rewards', color=(0., 0., 1., 0.3))
        plt.plot([float(d) for d in data['mean_train_reward']], label='Mean Rewards', color=(0., 0., 1., 1.))
        plt.title('Rewards per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.legend(loc="upper left")
        # plt.xscale('log')
        plt.show()
        
        plt.plot([int(p) for p in data['success_rate_ts']], [float(d) for d in data['success_rate']], color=(0., 0., 1., 1.))
        plt.title('Success Rate per Episode')
        plt.xlabel(f'Time steps')
        plt.ylabel('Success Rate')
        # plt.xscale('log')
        plt.show()


class DDPG_RandProp(object):
    def __init__(self, env, buffer, noise:str="OUNoise", theta:float=0.15, sigma:float=0.2, dt:float=0.1, folder:str='tmp_extended_ddpg/networks',
                 gamma:float=0.99, mem_size:int=int(1e6), burn_in_tsteps:int=int(25e3), batch_size:int=100, lr:float=0.001, polyak:float=0.1, upd_freq:int=2,
                 random_prop:float=0.3, nn_dims:list[int]=[256,256,256], env_id='PandaPickAndPlace-v3', n_envs:int=1, iteration:int=0, epsilon:float=1., bias:bool=True):
        self.folder = folder
        self.gamma = gamma
        self.env = env
        self.update_freq = upd_freq
        self.burn_in_tsteps = burn_in_tsteps
        self.random_prob = random_prop
        self.validate_timesteps = 0
        self.env_id = env_id
        # In order to force exploration use epsilon-greedy approach
        self.epsilon = epsilon
        self.epsion_decay = 0.999
        self.best_succ_rate = 0.0
        self.iteration = iteration + 1
        
        '''Panda-Gym V3'''
        self.single_env = gym.make(self.env_id, render=False)
        '''Panda-Gym V2'''
        # self.single_env = gym.make(self.env_id, control_type='ee', render=False)
        
        if noise == "OUNoise":
            self.noise = OUNoise(theta=theta, mu=np.zeros((n_envs, self.env.single_action_space.shape[-1])), sigma=sigma, dt=dt)
        else:
            self.noise = NormalNoise(sigma=sigma, size=self.env.single_action_space.shape[-1], n_envs=n_envs)
            
        state_shape = 0        
        if (isinstance(env.single_observation_space, gym.spaces.Dict)):
            for key, _obs_shape in env.single_observation_space.items():
                # if key == 'observation' or key == 'desired_goal':
                #     state_shape = state_shape + _obs_shape.shape[-1]          
                state_shape = state_shape + _obs_shape.shape[-1]  
        else:
            state_shape = env.single_observation_space.shape[-1]
        
        self.buffer = buffer
        '''stable-baselines3'''
        self.buffer.reset()
        self.batch_size = batch_size
        self.a_lb = env.single_action_space.low
        self.a_ub = env.single_action_space.high
        self.polyak = polyak
        
        self.update_loss = [] 
        self.update_critic_loss = []
        self.loss_evolution = [] # La pèrdua durant l'entrenament
        self.critic_loss_evolution = [] # La pèrdua durant l'entrenament
        self.training_rewards = [] # Les recompenses obtingudes a cada pas de l'entrenament
        self.mean_training_rewards = [] # Les recompenses mitjanes cada 100 episodis
        self.success_rate = [] # Average success rate on validaton model
        self.success_rate_ts = [] # Timestep which validation were performed
        
        '''Neural Networks for Panda Gym Environments'''
        self.actor = ActorNeuralNetwork(env=self.single_env, input_dims=state_shape, n_actions=env.single_action_space.shape[-1], lr=lr, nn_dims=nn_dims, name="Actor", 
                                        chkpt_dir=self.folder+'/networks', add_bias=bias)
        self.critic = CriticNeuralNetwork(env=self.single_env, input_dims=state_shape+env.single_action_space.shape[-1], name="Critic", n_outs=1, lr=lr, nn_dims=nn_dims, 
                                          chkpt_dir=self.folder+'/networks', add_bias=bias)
        self.target_actor = ActorNeuralNetwork(env=self.single_env, input_dims=state_shape, n_actions=env.single_action_space.shape[-1], lr=lr, nn_dims=nn_dims, 
                                               name="Target_Actor", chkpt_dir=self.folder+'/networks', add_bias=bias)
        self.target_critic = CriticNeuralNetwork(env=self.single_env, input_dims=state_shape+env.single_action_space.shape[-1], name="Target_Critic", n_outs=1, lr=lr, 
                                                 nn_dims=nn_dims, chkpt_dir=self.folder+'/networks', add_bias=bias)
        
        self.update_network_parameters(polyak=0) # polyak=1 to copy exactly the same params at the first time
        
    def get_action(self, state, evaluation:bool=False, random:bool=False):       
        if random:
            return np.array((self.env.action_space.sample() + 
                            self.noise.get_sample()).clip(self.a_lb, self.a_ub), dtype=float) # acció random
                
        if not evaluation:   
            if np.random.random() < self.epsilon: # self.random_prob:
                actions = np.array((self.env.action_space.sample() + 
                            self.noise.get_sample()).clip(self.a_lb, self.a_ub), dtype=float) # acció random  
            else:
                actions = (self.actor.forward(state).cpu().detach().data.numpy() + 
                            self.noise.get_sample()).clip(self.a_lb, self.a_ub)                     
        else:
            actions = [self.actor.forward(s).clip(self.a_lb, self.a_ub) for s in state]      
               
        return actions
      
    def train(self, max_tsteps=1000, update_steps=-1, tsteps_checkpoint=1000, load_from_checkpoint:bool=False, validate_timesteps:int=80, validate_eps:int=10):
        '''
        if update_steps <= 0 -> Update the networks at the end of each episode
        '''
        eps_rewards, steps = 0, 0
        self.episodes = 0
        self.validate_timesteps = validate_timesteps
        
        _load_from_checkpoint = load_from_checkpoint
        
        # Omplim el buffer amb N experiències aleatòries ()
        # if not _load_from_checkpoint:
        print("Filling replay buffer...")
        '''Panda-Gym V3'''
        state, info = self.env.reset()
        '''Panda-Gym V2'''
        # state = self.env.reset()
        # while np.linalg.norm(state['achieved_goal'] - state['desired_goal'], axis=-1) <= 0.05:
        #     state, info = self.env.reset()
        self.noise.reset()
        done = False
        timesteps = 0
        while timesteps < self.burn_in_tsteps:
            timesteps += 1
            
            action = self.get_action(state, random=True)
            
            '''Panda-Gym V3'''
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated | truncated
            '''Panda-Gym V2'''
            # next_state, reward, done, info = self.env.step(action)
            # any_success = np.array([i['is_success'] for i in info], dtype=bool)
            # done = done | any_success
           
            # self.buffer.append(state, action, reward, next_state, done)
            '''stable-baselines3'''
            self.buffer.add(obs=state, next_obs=next_state, action=action, reward=reward, done=done, infos=info, is_virtual=False)
            state = next_state.copy()                            
        
        print('Starting training...')
        if _load_from_checkpoint:
            self.load_checkpoint(folder=self.folder)
            max_tsteps = self.episodes
        
        done = False
        eps_rewards = 0
        eps_completed = 0
        '''Panda-Gym V3'''
        state, info = self.env.reset()
        '''Panda-Gym V2'''
        # state = self.env.reset()
        # while np.linalg.norm(state['achieved_goal'] - state['desired_goal'], axis=-1) < 0.07:
        #     state, info = self.env.reset()
        self.noise.reset() 
        for timesteps in tqdm(range(max_tsteps)):           
            steps += 1
            action = self.get_action(StateVecToList(state))  
            # action = self.get_action(state)  
            
            '''Panda-Gym V2'''
            # next_state, reward, done, info = self.env.step(action)
            # any_done = any(done)==True
            '''Validation done based on episodes completed'''
            # if any_done or any([i['is_success'] for i in info])==1.0:
            #     eps_completed += 1
            #     if eps_completed % validate_timesteps == 0:
            #         print('\t... Performing Validation of Model ...')
            #         self.success_rate.append(self.validate_model(n_eps=validate_eps))
            #         self.success_rate_ts.append(episode)
            #         print(f'\tSuccess Rate: {self.success_rate[-1]}')   
            
            '''Panda-Gym V3'''            
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated | truncated
            
            # Only update epsilon if the first environment has ended the episode
            if done[0]:
                # Actualitzar epsilon segons la velocitat de decaïment fixada
                self.epsilon = max(self.epsilon * self.epsion_decay, self.random_prob)
            
            '''Validation done based on episodes completed'''
            # if any(done)==True:
            #     eps_completed += 1
            #     if eps_completed % validate_timesteps == 0:
            #         print('\t... Performing Validation of Model ...')
            #         self.success_rate.append(self.validate_model(n_eps=validate_eps))
            #         self.success_rate_ts.append(timesteps)
            #         print(f'\tSuccess Rate: {self.success_rate[-1]}')
            '''Validation done based on timesteps achieved'''
            if (timesteps+1) % validate_timesteps == 0:
                print('\t... Performing Validation of Model ...')
                success_rate = self.validate_model(n_eps=validate_eps)
                if success_rate >= self.best_succ_rate:
                    self.best_succ_rate = success_rate
                    self.actor.save_checkpoint(folder='tmp/best_models', filename_header='DDPG_Iter_'+str(self.iteration))
                self.success_rate.append(success_rate)
                self.success_rate_ts.append(timesteps)
                print(f'\tSuccess Rate: {self.success_rate[-1]}')
            
            eps_rewards += reward
            
            # self.buffer.append(state, action, reward, next_state, done)
            '''stable-baselines3'''
            self.buffer.add(state, next_state, action, reward, done, info)
            self.update_networks(steps)
            state = next_state.copy()
            
            self.loss_evolution.append(np.mean(self.update_loss))
            self.critic_loss_evolution.append(np.mean(self.update_critic_loss))
            self.update_loss = []
            self.update_critic_loss = []
                       
            if timesteps % tsteps_checkpoint == 0 and timesteps > 1:
                self.episodes = max_tsteps - timesteps
                self.save_checkpoint()
                # print(f'Episode {episode} -> Success Rate: {self.success_rate[-1]*100.0} %')
        print('\t... Performing Final Validation of Model ...')
        success_rate = self.validate_model(n_eps=validate_eps)
        if success_rate >= self.best_succ_rate:
            self.best_succ_rate = success_rate
            self.actor.save_checkpoint(folder='tmp/best_models', filename_header='DDPG_Iter_'+str(self.iteration)+'_FinalModel')
        self.success_rate.append(success_rate)
        self.success_rate_ts.append(timesteps)
        print(f'\tSuccess Rate of Final Validation: {self.success_rate[-1]}')
                                
    def update_networks(self, timesteps, batch=None):
        # state, action, reward, next_state, done = self.buffer.sample_batch(batch_size=self.batch_size)
        '''stable-baselines3'''
        samples = self.buffer.sample(batch_size=self.batch_size, env=self.env)
        state = samples[0]
        action = samples[1]
        next_state = samples[2]
        done = samples[3]      
        reward = samples[4]
        
        # Separem les variables de l'experiència i les convertim a tensors
        if (isinstance(self.env.observation_space, gym.spaces.Dict)):
            # states = state
            # next_states = next_state
            states = StateVecToList(state)
            next_states = StateVecToList(next_state)
        else:
            states = np.array(state) #torch.tensor(np.array(state), dtype=torch.float).to(self.actor.device)
            next_states = np.array(next_state) #torch.tensor(np.array(next_state), dtype=torch.float).to(self.actor.device)
            
        # actions = torch.tensor(np.array(action), dtype=torch.float).to(self.actor.device)
        # rewards = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        # dones = torch.tensor(done, dtype=torch.float).to(self.actor.device)
        '''stable-baselines3'''
        actions = action
        rewards = reward
        dones = done
        
        with torch.no_grad():            
            next_action = (
                self.target_actor.forward(next_states).to(device=self.actor.device)
                ).clip(torch.tensor(self.a_lb).to(device=self.actor.device), torch.tensor(self.a_ub).to(device=self.actor.device))
            # Compute the target Q value
            target_Q = rewards + (1.0 - dones) * self.gamma * self.target_critic.forward(next_states, next_action)
           
        # Get current Q estimates
        current_Q = self.critic.forward(states, actions)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        
		# Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        
        if (timesteps % self.update_freq) == 0:
            # Update actor and target networks delayed        
            actor_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            
            self.update_loss.append(actor_loss.cpu().detach().numpy())
            self.update_critic_loss.append(critic_loss.cpu().detach().numpy())
            
            self.update_network_parameters()
        
    def update_network_parameters(self, polyak=None):
        if polyak is None:
            polyak = self.polyak
            
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        
        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)
        
        for name in critic_state_dict:
            critic_state_dict[name] = polyak*target_critic_state_dict[name].clone() + (1-polyak)*critic_state_dict[name].clone()
            
        for name in actor_state_dict:
            actor_state_dict[name] = polyak*target_actor_state_dict[name].clone() + (1-polyak)*actor_state_dict[name].clone()
            
        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        
    def validate_model(self, n_eps=100):
        success = 0
        self.actor.eval()
        for _ in range(n_eps):
            '''Panda-Gym V3'''
            state, info = self.single_env.reset()
            while np.linalg.norm(state['achieved_goal'] - state['desired_goal'], axis=-1) <= 0.05:
                state, info = self.single_env.reset()
            done = False
            while not done:
                action = self.actor.forward(state).cpu().detach().data.numpy()
                state, _, terminated, truncated, _ = self.single_env.step(action)                
                if terminated:
                      success += 1
                      done = True
                elif truncated:
                    done = True 
            '''Panda-Gym V2'''
            # state = self.single_env.reset()
            # while np.linalg.norm(state['achieved_goal'] - state['desired_goal'], axis=-1) <= 0.05:
            #     state = self.single_env.reset()
            # done = False            
            # while not done:
            #     action = self.actor.forward(state).cpu().detach().data.numpy()
            #     state, _, done, info = self.single_env.step(action)                
            #     if info['is_success']==1.0:
            #         success += 1
            #         done = True
                 
        self.actor.train()
        return 1.0*success/n_eps
       
    def save_models(self):
        self.actor.save_model(env_id=self.env_id)
        self.target_actor.save_model(env_id=self.env_id)
        self.critic.save_model(env_id=self.env_id)
        self.target_critic.save_model(env_id=self.env_id)
        
    def load_models(self, path='tmp_extended_ddpg\\networks', file_name=None):
        self.actor.load_model(path=path, file_name=file_name[0])
        self.target_actor.load_model(path=path, file_name=file_name[1])
        self.critic.load_model(path=path, file_name=file_name[2])
        self.target_critic.load_model(path=path, file_name=file_name[3])
        
    def save_checkpoint(self):  
        # fetch all files
        for file_name in os.listdir(self.folder+'/checkpoints/'):
            # construct full file path
            source = self.folder + '/checkpoints/' + file_name
            destination = self.folder + '/checkpoints_aux/' + file_name
            # copy only files
            if os.path.isfile(source):
                shutil.copy(source, destination)
                print('copied', file_name)            
        if not os.path.isdir(self.folder+'/checkpoints'):
            print('... Creating checkpoint folder ...')
            os.mkdir(self.folder+'/checkpoints')
        
        print('... Saving chekcpoint ....')
        self.actor.save_checkpoint(folder=self.folder+'/checkpoints', filename_header='')
        self.target_actor.save_checkpoint(folder=self.folder+'/checkpoints', filename_header='')
        self.critic.save_checkpoint(folder=self.folder+'/checkpoints', filename_header='')
        self.target_critic.save_checkpoint(folder=self.folder+'/checkpoints', filename_header='')
        # self.buffer.save_checkpoint(folder=self.folder+'/checkpoints')
        
        print('... Saving training information ...')
        dict_params = {"mean_loss": np.array(self.loss_evolution).tolist(), 
                       "train_reward": self.training_rewards, 
                       "mean_train_reward": self.mean_training_rewards,
                       "mean_critic_loss": np.array(self.critic_loss_evolution).tolist(),
                       "success_rate": self.success_rate,
                       "success_rate_ts": self.success_rate_ts,
                       "actor_filename": '_' + self.actor.file_name,
                       "target_actor_filename": '_' + self.target_actor.file_name,
                       "critic_filename": '_' + self.critic.file_name,
                       "target_critic_filename": '_' + self.target_critic.file_name,
                       "episodes": self.episodes 
                       }
        
        with open(self.folder+'/checkpoints/ddpg_train_info.json', 'w') as wf:
            json.dump(dict_params, wf)
    
    def load_checkpoint(self, folder='tmp_extended_ddpg'):
        print('... Loading values from checkpoint ...')
        with open(folder+'/checkpoints/ddpg_train_info.json') as rf:
            dict_res = json.load(rf)
        
        print('mean loss ...')
        self.loss_evolution = dict_res['mean_loss']
        print('train reward ...')
        self.training_rewards = dict_res['train_reward'] 
        print('mean train reward ...')
        self.mean_training_rewards = dict_res['mean_train_reward'] 
        print('mean critic loss ...')
        self.critic_loss_evolution = dict_res['mean_critic_loss'] 
        print('success rate ...')
        self.success_rate = dict_res['success_rate'] 
        print('success rate timesteps ...')
        self.success_rate_ts = dict_res['success_rate_ts'] 
        print('episodes ...')
        self.episodes = dict_res['episodes']
        
        print('Neural Nets ...')
        self.actor.load_checkpoint(dict_res['actor_filename'], folder=folder)
        self.target_actor.load_checkpoint(dict_res['target_actor_filename'], folder=folder)
        self.critic.load_checkpoint(dict_res['critic_filename'], folder=folder)
        self.target_critic.load_checkpoint(dict_res['target_critic_filename'], folder=folder)
        
        # print('Buffer ...')
        # self.buffer.load_checkpoint(folder=folder)
        
    def save_results(self, folder='tmp_extended_ddpg'):
        dict_params = {"mean_loss": [str(el) for el in self.loss_evolution],
                       "train_reward": [str(el) for el in self.training_rewards],
                       "mean_train_reward": [str(el) for el in self.mean_training_rewards],
                       "mean_critic_loss": [str(el) for el in self.critic_loss_evolution],
                       "success_rate": [str(el) for el in self.success_rate],
                       "success_rate_ts": self.success_rate_ts
                       }
        
        date_now = time.strftime("%Y%m%d%H%M")
        with open(folder+'/train_res/'+date_now+'_'+self.env_id+'_ddpg_train_vals.json', 'w') as wf:
            json.dump(dict_params, wf)        
            
    def load_results(self, file):
        with open(file) as rf:
            dict_res = json.load(rf)
        return dict_res
    
    def plot_results(self, file='tmp_extended_ddpg/train_res/ddpg_train_vals.json'):
        data = self.load_results(file)
        
        # Grafico la posició final, l'evolució mitjana de les pèrdues i l'evolució d'epsilon
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1)

        ax1 = plt.subplot(gs[0,:])
        ax1.plot([float(d) for d in data['mean_loss']], label='Mean Actor Loss', color=(0., 0., 1., 1.))
        ax1.plot([float(d) for d in data['mean_critic_loss']], label='Mean Critic Loss', color=(1., 0., 0., 0.6))        
        ax1.set_title('Mean Loss per Episode')
        ax1.set_xlabel('Time steps')
        ax1.set_ylabel('Loss')
        # ax1.set_xscale('log')
        ax1.legend(loc="upper left")

        ax2 = plt.subplot(gs[1,:])
        ax2.plot([float(d) for d in data['train_reward']], label='Rewards', color=(0., 0., 1., 0.3))
        ax2.plot([float(d) for d in data['mean_train_reward']], label='Mean Rewards', color=(0., 0., 1., 1.))
        ax2.set_title('Rewards per Episode')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Rewards')
        # ax2.set_xscale('log')
        ax2.legend(loc="upper left")
        
        fig.add_subplot(ax1)
        fig.add_subplot(ax2)
        
        plt.show()
        
        plt.plot([float(d) for d in data['mean_loss']], label='Mean Actor Loss', color=(0., 0., 1., 1.))
        plt.plot([float(d) for d in data['mean_critic_loss']], label='Mean Critic Loss', color=(1., 0., 0., 0.6))
        plt.title('Mean Loss per Episode')
        plt.xlabel('Time steps')
        plt.ylabel('Loss')
        plt.legend(loc="upper left")
        # plt.xscale('log')
        plt.show()

        plt.plot([float(d) for d in data['train_reward']], label='Rewards', color=(0., 0., 1., 0.3))
        plt.plot([float(d) for d in data['mean_train_reward']], label='Mean Rewards', color=(0., 0., 1., 1.))
        plt.title('Rewards per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.legend(loc="upper left")
        # plt.xscale('log')
        plt.show()
        
        plt.plot([int(p) for p in data['success_rate_ts']], [float(d) for d in data['success_rate']], color=(0., 0., 1., 1.))
        plt.title('Success Rate per Episode')
        plt.xlabel(f'Time steps')
        plt.ylabel('Success Rate')
        # plt.xscale('log')
        plt.show()



if __name__ == '__main__':
    n_envs = 8
    '''Panda-Gym V3'''
    env_id = 'PandaPickAndPlace-v3'
    # env_id = 'PandaReach-v3'
    # env_id = 'PandaPush-v3'
    # env_id = 'PandaSlide-v3'
    envs = gym.vector.make(env_id, num_envs=n_envs, render=False, asynchronous=True)
    '''Panda-Gym V2'''
    # env_id = 'PandaReach-v2'
    # env_id = 'PandaPush-v2'
    # env_id = 'PandaSlide-v2'
    # envs = gym.vector.make(env_id, control_type='ee', num_envs=n_envs, render=False)
    
    buffer_type = "HER"    
    agent_type = "DDPG" # TD3
    
    s_folder = 'tmp'
    lr = 1e-3
    nn_dims = [256, 256, 256]    
    mem_size = int(1e6) 
    burn_in_tsteps = int(1e5)
    
    for i in range(7):
        if i<2:
            agent_type = "DDPG"
        else:
            agent_type = "TD3"
        
        if buffer_type == "HER":
            # buffer = HERBuffer(envs, memory_size=mem_size, env_id=env_id, n_sampled_goal=4)
            '''stable-baselines3'''
            buffer = HerReplayBuffer(buffer_size=mem_size, observation_space=envs.single_observation_space, action_space=envs.single_action_space,
                                     env=envs, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), n_envs=n_envs,
                                     optimize_memory_usage=False, handle_timeout_termination=False, n_sampled_goal=4, goal_selection_strategy="future",
                                     online_sampling=True, env_id=env_id)            
        else:
            buffer = experienceReplayBuffer(envs, memory_size=mem_size, burn_in=burn_in_tsteps) # buffer de repeticions
                 
        if agent_type.upper() == "DDPG":
            agent = DDPG_RandProp(envs, buffer, noise="Normal", theta=0.15, sigma=0.2, dt=0.1, folder=s_folder, gamma=0.99, mem_size=mem_size,
                                burn_in_tsteps=burn_in_tsteps, batch_size=256, lr=lr, polyak=0.95, upd_freq=1, random_prop=0.3, nn_dims=nn_dims,
                                env_id=env_id, n_envs=n_envs, iteration=i+2, epsilon=0.3, bias=True)
        elif agent_type.upper() == "TD3":
            agent = TD3_RandProp(envs, buffer, noise="Normal", theta=0.15, sigma=0.2, dt=0.1, folder=s_folder, gamma=0.99, mem_size=mem_size,
                                burn_in_tsteps=burn_in_tsteps, batch_size=256, lr=lr, polyak=0.95, upd_freq=2, random_prop=0.3, nn_dims=nn_dims,
                                env_id=env_id, n_envs=n_envs, iteration=abs(i-3), epsilon=0.3, bias=True)
    
        agent.train(max_tsteps=int(1.15e6), update_steps=1, tsteps_checkpoint=20000, load_from_checkpoint=False, validate_timesteps=1000, validate_eps=80)
        agent.save_models()
        agent.save_results(folder=s_folder)

    envs.close()

