import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Union
import os
import time
from copy import deepcopy
import gymnasium as gym


class CriticNeuralNetwork(nn.Module):
    def __init__(self, env, input_dims, name, optimizer=None, n_outs=1, lr=1e-3, chkpt_dir='tmp_extended/networks', nn_dims=[256, 256, 256],  add_bias:bool=True):
        '''
        CriticNeuralNetwork
        ==================
        Class to define the network and behaviour of the actor neural network
        :param env: environment which will be used
        :param input_dims: number of the input dimensions for the first layer
        :param name: name to be used when saving the model
        :param optimizer: if you want to use a different optimizer than the one which is defined here
        :param n_outs: number of dimensions for the output of last layer
        :param lr: learning rate for the default optimizer
        :param chkpt_dir: the folder to save the model during training
        :param nn_dims: dimensions for the three layers used in this model
        :param bias: tag to remove bias or not from the layers (used for testing)
        '''
        super(CriticNeuralNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.n_outs = n_outs
        self.checkpoint_file = chkpt_dir 
        self.file_name = name + '.pth'
        self.env = env        
                
        self.layer1 = nn.Linear(self.input_dims, nn_dims[0], bias=add_bias)        
        self.bn1 = nn.LayerNorm(nn_dims[0])  
        self.layer2 = nn.Linear(nn_dims[0], nn_dims[1], bias=add_bias)        
        self.bn2 = nn.LayerNorm(nn_dims[1])  
        self.layer3 = nn.Linear(nn_dims[1], nn_dims[2], bias=add_bias)        
        self.bn3 = nn.LayerNorm(nn_dims[2])  
        # Critic layer
        self.q = nn.Linear(nn_dims[2], 1, bias=add_bias)       
        
        # If we have a CUDA device we allow to work with it
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # Define the optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
        
        # Initialize the weights and bias to a custom range
        self.reset_parameters()
        
    def hidden_init(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.layer1.weight.data, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layer2.weight.data, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layer3.weight.data, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.q.weight.data, mode='fan_in', nonlinearity='relu')
        
    def forward(self, state, action:torch.FloatTensor) -> torch.Tensor:
        if (isinstance(state, dict)):
            # If the state is Dict prepare the data before feed it to the layer
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
                x = torch.FloatTensor(_state).to(device=self.device) 
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
                x = torch.FloatTensor(_state).to(device=self.device) 
            x = torch.cat((x, action.squeeze(1)), dim=1)
        else:
            state = torch.FloatTensor(state).to(device=self.device)
            x = torch.cat((state, action), dim=1).to(device=self.device)
        state_value = self.layer1(x)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.layer2(state_value)
        state_value = self.bn2(state_value)
        state_value = F.relu(state_value)
        state_value = self.layer3(state_value)
        state_value = self.bn3(state_value)
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
