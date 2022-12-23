import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, OrderedDict, Union
import os
import time
from copy import deepcopy
# import gym
import gymnasium as gym


class ActorNeuralNetwork(nn.Module):
    def __init__(self, env, input_dims, n_actions, name, optimizer=None, lr=1e-3, chkpt_dir='tmp_extended/networks', nn_dims=[256, 256, 256], epsilon=1e-8,
                 add_bias:bool=True):
        super(ActorNeuralNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.checkpoint_file = chkpt_dir 
        self.file_name = name + '.pth'
        self.lr = lr
        self.nn_dims = nn_dims
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
        
        # self.bn0 = nn.LayerNorm(self.input_dims)
        self.layer1 = nn.Linear(self.input_dims, nn_dims[0], bias=add_bias)
        self.bn1 = nn.LayerNorm(nn_dims[0])  
        self.layer2 = nn.Linear(nn_dims[0], nn_dims[1], bias=add_bias)                 
        self.bn2 = nn.LayerNorm(nn_dims[1])  
        self.layer3 = nn.Linear(nn_dims[1], nn_dims[2], bias=add_bias)                 
        self.bn3 = nn.LayerNorm(nn_dims[2])  
        self.action = nn.Linear(nn_dims[2], n_actions, bias=add_bias)       
        
        # If we have a CUDA device we allow to work with it
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # Define the optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
            
        self.reset_parameters()
            
    def hidden_init(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)

    def reset_parameters(self):
        # DEFAULT: Wl1=(-0.0625, 0.0625); Wl2=(-0.0625, 0.0625); Wl3=(-0.0625, 0.0625); Wq=(-3e3, 3e3)
        # nn.init.uniform_(self.layer1.weight.data, *self.hidden_init(self.layer1))
        nn.init.uniform_(self.layer1.weight.data, -0.07, 0.07) # -0.08, 0.08)
        nn.init.uniform_(self.layer1.bias.data, -0.65, 0.65) # -0.8, 0.8)
        # nn.init.uniform_(self.layer2.weight.data, *self.hidden_init(self.layer2))
        nn.init.uniform_(self.layer2.weight.data, -0.07, 0.07) # -0.08, 0.08)
        nn.init.uniform_(self.layer2.bias.data, -0.65, 0.65) # -0.8, 0.8)
        # nn.init.uniform_(self.layer3.weight.data, *self.hidden_init(self.layer3))
        nn.init.uniform_(self.layer3.weight.data, -0.07, 0.07) # -0.08, 0.08)
        nn.init.uniform_(self.layer3.bias.data, -0.65, 0.65) # -0.8, 0.8)
        nn.init.uniform_(self.action.weight.data, -0.003, 0.003) #-3.5e-3, 3.5e-3)
        nn.init.uniform_(self.action.bias.data, -0.004, 0.004) #  -0.01, 0.01)
     
    def forward(self, state) -> torch.Tensor:
        if (isinstance(state, Union[Dict, OrderedDict, gym.spaces.Dict])):
            if isinstance(state["observation"], torch.Tensor):
                ach_list, obs_list, des_list = [], [], []
                for k,v in state.items():
                    if k=='achieved_goal':
                        ach_list.append(v.cpu().detach().numpy())
                    if k == 'observation':
                        obs_list.append(v.cpu().detach().numpy())
                    elif k == 'desired_goal':
                        des_list.append(v.cpu().detach().numpy())
                _state = []
                for a,o,d in zip(ach_list,obs_list, des_list):
                    _state.append([*a,*o,*d])  
                _state = np.array(_state).squeeze()   
            else:
                ach_list, obs_list, des_list = [], [], []
                for k,v in state.items():
                    if k=='achieved_goal':
                        ach_list.append(v)
                    if k == 'observation':
                        obs_list.append(v)
                    elif k == 'desired_goal':
                        des_list.append(v)
                _state = []
                for a,o,d in zip(ach_list,obs_list, des_list):
                    _state.append([*a,*o,*d])  
                
                _state = np.array(_state).squeeze()              
        else:
            _state = state.copy()
            # x = torch.FloatTensor(state).to(device=self.device)
        x = torch.FloatTensor(_state).to(device=self.device)
        # x = self.bn0(x)
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = F.relu(x)     
        
        return torch.tanh(self.action(x))    
   
    def save_checkpoint(self, folder, filename_header):
        torch.save(self.state_dict(), os.path.join(folder, filename_header+'_'+self.file_name))
    
    def load_checkpoint(self, file_name, folder='tmp_extended'):
        self.load_state_dict(torch.load(os.path.join(folder+'/checkpoints', file_name)))
    
    def save_model(self, env_id):
        print('... Saving checkpoint ...')
        date_now = time.strftime("%Y%m%d%H%M")
        torch.save(self.state_dict(), os.path.join(self.checkpoint_file, date_now+'_'+env_id+'_'+self.file_name))
        
    def load_model(self, path=None, file_name=None):
        print('... Loading checkpoint ...')
        if path is None or file_name is None:
            if torch.cuda.is_available():
                self.load_state_dict(torch.load(self.checkpoint_file))
            else:
                self.load_state_dict(torch.load(self.checkpoint_file, map_location=torch.device('cpu')))
        else:
            if torch.cuda.is_available():
                self.load_state_dict(torch.load(os.path.join(path, file_name)))
            else:
                self.load_state_dict(torch.load(os.path.join(path, file_name), map_location=torch.device('cpu')))
  
class ActorNeuralNetworkLayerNorm(nn.Module):
    def __init__(self, input_dims, n_actions, name, lr=1e-3, chkpt_dir='tmp_extended/networks', nn_dims=[256, 256, 256]):
        super(ActorNeuralNetworkLayerNorm, self).__init__()
        
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.checkpoint_file = chkpt_dir 
        self.file_name = name + '.pth'
        
        self.batch0 = nn.LayerNorm(self.input_dims)
        self.layer1 = nn.Linear(self.input_dims, nn_dims[0])         
        self.batch1 = nn.LayerNorm(nn_dims[0])
        self.layer2 = nn.Linear(nn_dims[0], nn_dims[1])                 
        self.batch2 = nn.LayerNorm(nn_dims[1])
        self.layer3 = nn.Linear(nn_dims[1], nn_dims[2])
        self.batch3 = nn.LayerNorm(nn_dims[2])
        self.action = nn.Linear(nn_dims[2], n_actions)        
        
        # If we have a CUDA device we allow to work with it
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, state) -> torch.Tensor:
        if (isinstance(state, dict)):
            _state = []
            for key in sorted(state.keys(), key=lambda x:x.lower()):
                _state.append(state[key])
            x = torch.FloatTensor(np.array([item for sublist in _state for item in sublist])).to(device=self.device)
        else:
            x = torch.FloatTensor(state).to(device=self.device)
        x = self.batch0(x)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.batch1(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.batch2(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.batch3(x)
        
        return torch.tanh(self.action(x))    
   
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
            if torch.cuda.is_available():
                self.load_state_dict(torch.load(self.checkpoint_file))
            else:
                self.load_state_dict(torch.load(self.checkpoint_file, map_location=torch.device('cpu')))
        else:
            if torch.cuda.is_available():
                self.load_state_dict(torch.load(os.path.join(path, file_name)))
            else:
                self.load_state_dict(torch.load(os.path.join(path, file_name), map_location=torch.device('cpu')))
                  