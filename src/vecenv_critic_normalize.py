import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Union
import os
import time
from copy import deepcopy
import gym
# import gymnasium as gym


class CriticNeuralNetwork(nn.Module):
    def __init__(self, env, input_dims, name, optimizer=None, n_outs=1, lr=1e-3, chkpt_dir='tmp_extended/networks', nn_dims=[256, 256, 256], epsilon=1e-8):
        super(CriticNeuralNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.n_outs = n_outs
        self.checkpoint_file = chkpt_dir 
        self.file_name = name + '.pth'
        self.env = env        
        self.epsilon = epsilon        
        self.obs_max = max(max(self.env.observation_space['observation'].high),
                           max(self.env.observation_space['achieved_goal'].high),
                           max(self.env.observation_space['desired_goal'].high))
        
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            self.obs_spaces = self.env.observation_space.spaces
            self.obs_keys = [k for k,_ in self.env.observation_space.items()]
            self.mean = {key: np.zeros(shape=self.obs_spaces[key].shape) for key in [k for k in self.obs_keys]}
            self.var = {key: np.zeros(shape=self.obs_spaces[key].shape) for key in [k for k in self.obs_keys]}
            self.count = {key: 0 for key in [k for k in self.obs_keys]}
        else:
            self.obs_spaces = None
            self.obs_keys = None
            self.mean = np.zeros(shape=self.obs_spaces.shape)
            self.var = np.zeros(shape=self.obs_spaces.shape)                
            self.count = 0
        
        self.layer1 = nn.Linear(self.input_dims, nn_dims[0])        
        self.layer2 = nn.Linear(nn_dims[0], nn_dims[1])        
        self.layer3 = nn.Linear(nn_dims[1], nn_dims[2])        
        # Critic layer
        self.q = nn.Linear(nn_dims[2], 1)        
        
        # If we have a CUDA device we allow to work with it
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # Define the optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
            
    def update_statistics(self, arr) -> None:
        if self.obs_keys is not None:
            batch_mean = {key: 0. for key in [k for k in self.obs_keys]}
            batch_var = {key: 0. for key in [k for k in self.obs_keys]}
            batch_count = {key: 0 for key in [k for k in self.obs_keys]}
            for k in self.obs_keys:
                batch_mean[k] = np.mean(arr[k].flatten().cpu().detach().data.numpy(), axis=0)
                batch_var[k] = np.var(arr[k].flatten().cpu().detach().data.numpy(), axis=0)
                batch_count[k] = arr[k].flatten().cpu().detach().data.numpy().shape[0]
        else:
            batch_mean = np.mean(arr, axis=0)
            batch_var = np.var(arr, axis=0)
            batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: Union[int, float, Dict]) -> None:
        if self.obs_keys is not None:
            for k in self.obs_keys:
                delta = batch_mean[k] - self.mean[k]
                tot_count = self.count[k] + batch_count[k]

                new_mean = self.mean[k] + delta * batch_count[k] / tot_count
                m_a = self.var[k] * self.count[k]
                m_b = batch_var[k] * batch_count[k]
                m_2 = m_a + m_b + np.square(delta) * self.count[k] * batch_count[k] / (self.count[k] + batch_count[k])
                new_var = m_2 / (self.count[k] + batch_count[k])

                new_count = batch_count[k] + self.count[k]

                self.mean[k] = new_mean
                self.var[k] = new_var
                self.count[k] = new_count
        else:
            delta = batch_mean - self.mean
            tot_count = self.count + batch_count

            new_mean = self.mean + delta * batch_count / tot_count
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
            new_var = m_2 / (self.count + batch_count)

            new_count = batch_count + self.count

            self.mean = new_mean
            self.var = new_var
            self.count = new_count
        
    def _normalize_obs(self, obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        Helper to normalize observation.
        :param obs:
        :param obs_rms: associated statistics
        :return: normalized observation
        """
        return np.clip((obs - mean) / np.sqrt(std + self.epsilon), -self.obs_max, self.obs_max)
       
        
    def forward(self, state, action:torch.FloatTensor) -> torch.Tensor:
        if (isinstance(state, dict)):
            if isinstance(state["observation"], torch.Tensor):
                self.update_statistics(state)
                _state_ = deepcopy(state)
                # for k,_ in state.items():
                #     _state_[k] = self._normalize_obs(_state_[k], self.mean[k], self.var[k])
                # obs = state['observation'].cpu().detach().data.numpy()
                # des = state['desired_goal'].cpu().detach().data.numpy()
                obs = _state_['observation'].cpu().detach().data.numpy()
                des = _state_['desired_goal'].cpu().detach().data.numpy()
                x = torch.FloatTensor(np.array([np.concatenate([o, d]) for o,d in zip(obs,des)])).to(device=self.device)
                # x = torch.FloatTensor(np.array([np.concatenate([o, d], axis=1) for o,d in zip(obs,des)])).to(device=self.device)
            else:
                x = torch.FloatTensor(np.concatenate([state['observation'], state['desired_goal']])).to(device=self.device)
                # _state = []
                # for key in sorted(state.keys(), key=lambda x:x.lower()):
                #     _state.append(state[key])
                # x = torch.FloatTensor(np.array([item for sublist in _state for item in sublist])).to(device=self.device)
            x = torch.cat((x.squeeze(1), action.squeeze(1)), dim=1)
        else:
            state = torch.FloatTensor(state).to(device=self.device)
            x = torch.cat((state, action), dim=1).to(device=self.device)
        state_value = self.layer1(x)
        state_value = F.relu(state_value)
        state_value = self.layer2(state_value)
        state_value = F.relu(state_value)
        state_value = self.layer3(state_value)
        state_value = F.relu(state_value)
        
        return self.q(state_value)    
    
    def save_checkpoint(self, folder, filename_header):
        torch.save(self.state_dict(), os.path.join(folder, filename_header+'_'+self.file_name))
    
    def load_checkpoint(self, file_name, folder='tmp_extended'):
        self.load_state_dict(torch.load(os.path.join(folder+'/checkpoints', file_name)))
    
    def save_model(self, env_id):
        print('... Saving checkpoint ...')
        date_now = time.strftime("%Y%m%d%H%M")
        torch.save(self.state_dict(), os.path.join(self.checkpoint_file, date_now+''+env_id+'_'+self.file_name))
        
    def load_model(self, path=None, file_name=None):
        print('... Loading checkpoint ...')
        if path is None or file_name is None:
            if(torch.cuda.is_available()):
                self.load_state_dict(torch.load(self.checkpoint_file))
            else:
            ### LOAD WITHOUT CUDA
                self.load_state_dict(torch.load(self.checkpoint_file, map_location=torch.device('cpu')))
        else:
            if(torch.cuda.is_available()):
                self.load_state_dict(torch.load(os.path.join(path, file_name)))
            else:
                ### LOAD WITHOUT CUDA
                self.load_state_dict(torch.load(os.path.join(path, file_name), map_location=torch.device('cpu')))


class CriticNeuralNetworkLayerNorm(nn.Module):
    def __init__(self, input_dims, name, n_outs=1, lr=1e-3, chkpt_dir='tmp_extended/networks', nn_dims=[256, 256, 256]):
        super(CriticNeuralNetworkLayerNorm, self).__init__()
        
        self.input_dims = input_dims
        self.n_outs = n_outs
        self.checkpoint_file = chkpt_dir 
        self.file_name = name + '.pth'
        
        self.batch0 = nn.LayerNorm(self.input_dims)
        self.layer1 = nn.Linear(self.input_dims, nn_dims[0])         
        self.batch1 = nn.LayerNorm(nn_dims[0])
        self.layer2 = nn.Linear(nn_dims[0], nn_dims[1])                 
        self.batch2 = nn.LayerNorm(nn_dims[1])
        self.layer3 = nn.Linear(nn_dims[1], nn_dims[2])
        self.batch3 = nn.LayerNorm(nn_dims[2])
        
        # Critic layer
        self.q = nn.Linear(nn_dims[2], 1)        
        
        # If we have a CUDA device we allow to work with it
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.01)
        
    def forward(self, state, action:torch.FloatTensor) -> torch.Tensor:
        if (isinstance(state, dict)):
            _state = []
            for key in sorted(state.keys(), key=lambda x:x.lower()):
                _state.append(state[key])
            x = torch.FloatTensor(np.array([item for sublist in _state for item in sublist])).to(device=self.device)
            x = torch.cat((x, action))
        else:
            state = torch.FloatTensor(state).to(device=self.device)
            x = torch.cat((state, action), dim=1).to(device=self.device)
        state_value = self.batch0(x)
        state_value = self.layer1(state_value)
        state_value = F.relu(state_value)
        state_value = self.batch1(state_value)
        state_value = self.layer2(state_value)
        state_value = F.relu(state_value)
        state_value = self.batch2(state_value)
        state_value = self.layer3(state_value)
        state_value = F.relu(state_value)
        state_value = self.batch3(state_value)
                
        return self.q(state_value)    
    
    def save_checkpoint(self, folder, filename_header):
        torch.save(self.state_dict(), os.path.join(folder, filename_header+'_'+self.file_name))
    
    def load_checkpoint(self, file_name, folder='tmp_extended'):
        self.load_state_dict(torch.load(os.path.join(folder+'/checkpoints', file_name)))
    
    def save_model(self):
        print('... Saving checkpoint ...')
        date_now = time.strftime("%Y%m%d%H%M")
        torch.save(self.state_dict(), os.path.join(self.checkpoint_file, date_now+'_'+self.file_name))
        
    def load_model(self, path=None, file_name=None):
        print('... Loading checkpoint ...')
        if path is None or file_name is None:
            if(torch.cuda.is_available()):
                self.load_state_dict(torch.load(self.checkpoint_file))
            else:
            ### LOAD WITHOUT CUDA
                self.load_state_dict(torch.load(self.checkpoint_file, map_location=torch.device('cpu')))
        else:
            if(torch.cuda.is_available()):
                self.load_state_dict(torch.load(os.path.join(path, file_name)))
            else:
                ### LOAD WITHOUT CUDA
                self.load_state_dict(torch.load(os.path.join(path, file_name), map_location=torch.device('cpu')))
