import torch.nn as nn
import torch
import numpy as np
from torch.nn import init
import os
from config import vehicleDynamicConfig

class Actor(nn.Module):
    def __init__(self, inputSize, outputSize, lr=0.001):
        super().__init__()
        self._out_gain = torch.FloatTensor([2.0, 0.1])
        # self._norm_matrix = 1 * \
        #     torch.tensor([1, 1, 1, 1], dtype=torch.float32)
        self._norm_matrix = torch.ones(inputSize, dtype=torch.float32)
        
        self.layers = nn.Sequential(
            nn.Linear(inputSize, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, outputSize),
            nn.Tanh()
        )

        # optimizer
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size = 1000, gamma=0.95, last_epoch=-1)
        self._initializeWeights()

    def forward(self, x):
        temp = x * self._norm_matrix
        x = self._out_gain * self.layers(temp)
        return x

    def predict(self, x):
        return self.forward(x).detach().numpy()

    def saveParameters(self, logdir):
        torch.save(self.state_dict(), os.path.join(logdir, "actor.pth"))

    def loadParameters(self, load_dir):
        self.load_state_dict(torch.load(os.path.join(load_dir, 'actor.pth')))

    def _initializeWeights(self):
        """
        initial parameter using xavier
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init.xavier_uniform_(m.weight)
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0.0)

        for name, module in self.layers.named_children():
            if name in ['10']: # 将倒数第一层的权重设为0，网络正常训练
                module.weight.data = module.weight.data * 0.0001
                # module.bias.data = torch.zeros_like(module.bias)


class ActorForIDC(Actor):
    def __init__(self, inputSize, outputSize, lr=0.001):
        super().__init__(inputSize, outputSize, lr)
        config = vehicleDynamicConfig()
        self.refNum = config.refNum
        self.inputSize = inputSize 

    def forward(self, x):
        x = self.preprocess(x)
        # x for NN: [u, v, w, [delta_x, delta_y, delta_phi] * N]
        x = super().forward(x)
        x = x/self._out_gain
        result = torch.zeros_like(x)
        result[:, 0] = x[:, 1]
        result[:, 1] = x[:, 0]
        return result
    
    def preprocess(self, obs):
        x = torch.zeros([1, self.inputSize])
        x[:, :3] = obs[:, :3]
        obs_ego = torch.concat([obs[:, 3:6], torch.zeros(1, 1)], dim = 1) 
        tempState = obs[:, 10: 10+4*self.refNum] - obs_ego.repeat(1, self.refNum)
        for i in range(self.refNum):
            relIndex = 4 * i + 3
            tempIndex = 4 * i
            x[:, relIndex] = tempState[:, tempIndex] * torch.cos(obs[:, 5]*np.pi/180) + tempState[:, tempIndex+1] * torch.sin(obs[:, 5]*np.pi/180)
            x[:, relIndex + 1] = tempState[:, tempIndex] * (-torch.sin(obs[:, 5]*np.pi/180)) + tempState[:, tempIndex+1] *  torch.cos(obs[:, 5]*np.pi/180)
            x[:, relIndex + 2] =  (tempState[:, tempIndex + 2]*np.pi/180)
            x[:, relIndex + 3] = torch.sin(tempState[:, tempIndex + 2]*np.pi/180)
        return x

class Critic(nn.Module):
    def __init__(self, inputSize, outputSize, lr=0.001):
        super().__init__()
        # initial parameters of actor
        self.layers = nn.Sequential(
            nn.Linear(inputSize, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, outputSize),
        )
        self._norm_matrix = torch.ones(inputSize, dtype=torch.float32)
        # initial optimizer
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, 1000, gamma=0.95, last_epoch=-1)
        self._initializeWeights()

    def forward(self, x):
        x = torch.mul(x, self._norm_matrix)
        x = self.layers(x)
        return x.reshape(x.size(0))

    def predict(self, x):
        return self.forward(x).detach().numpy()

    def saveParameters(self, logdir):
        torch.save(self.state_dict(), os.path.join(logdir, "critic.pth"))

    def loadParameters(self, load_dir):
        self.load_state_dict(torch.load(os.path.join(load_dir, 'critic.pth')))

    def _initializeWeights(self):
        """
        initial paramete using xavier
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0.0)
        for name, module in self.layers.named_children():
            if name in ['6']: # 将倒数第一层的权重设为0，网络正常训练
                module.weight.data = module.weight.data * 0.0001
                # module.bias.data = torch.zeros_like(module.bias)