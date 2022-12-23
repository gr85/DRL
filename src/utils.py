import numpy as np
import torch



def StateVecToList(observations):
    ach_list, obs_list, des_list = [], [], []
    for k,v in observations.items():
        if k=='achieved_goal':
            if isinstance(v, torch.Tensor):
                ach_list.append(v.cpu().detach().numpy())
            else:
                ach_list.append(v)
        if k == 'observation':
            if isinstance(v, torch.Tensor):
                obs_list.append(v.cpu().detach().numpy())
            else:
                obs_list.append(v)
        elif k == 'desired_goal':
            if isinstance(v, torch.Tensor):
                des_list.append(v.cpu().detach().numpy())
            else:
                des_list.append(v)
    states = []
    for a,o,d in zip(ach_list[0],obs_list[0], des_list[0]):
        states.append([*a,*o,*d])
    
    return states

