import numpy as np
import torch

class NoiseBase(object):
    def __init__(self):
        pass
    
    def reset(self):
        pass

class OUNoise(NoiseBase):
    '''
    Ornstein-Uhlenbeck noise 
    ========================
    Noise to be added with the action took by the agent
    in order to force the exploration of the action space
    '''
    def __init__(self, theta:float=0.15, mu:np.array=np.zeros(shape=1, dtype=float), sigma:float=0.2, dt:float=1e-1, x0=None):
        super(OUNoise, self).__init__()
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt        
        self.x0 = x0
        self.ll = np.array(-0.5, like=self.mu, dtype=float)
        self.hl = np.array(0.5, like=self.mu, dtype=float)
        
        self.reset()
                
    def get_sample(self) -> np.array:
        x = self.x_prev + self.theta*(self.mu - self.x_prev)*self.dt + self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        
        return torch.FloatTensor(x).clamp(torch.FloatTensor(self.ll), torch.FloatTensor(self.hl)).cpu().detach().data.numpy()
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        

class NormalNoise(object):
    def __init__(self, sigma:float=0.2, size:int=4, mu:float=0.0, n_envs:int=1):
        super(NormalNoise, self).__init__()
        self.sigma = sigma
        self.mu = mu
        self.size = size
        self.n_envs = n_envs
        self.ll = np.full((self.n_envs, self.size),-0.5, dtype=float)
        self.hl = np.full((self.n_envs, self.size), 0.5, dtype=float)
        
    def get_sample(self) -> np.array:
        return (torch.randn((self.n_envs, self.size)) * self.sigma + self.mu).clamp(torch.FloatTensor(self.ll), torch.FloatTensor(self.hl)).cpu().detach().data.numpy()
    
    def reset(self):
        pass
    
    
    