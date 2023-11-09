import torch.nn as nn
import torch
import numpy as np
from torch.nn import init
import os
from config import vehicleDynamic

class Actor(nn.Module):
    def __init__(self, inputSize, outputSize, lr=0.001):
        super().__init__()
        self._out_gain = torch.FloatTensor([2.0, 0.3])
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
        temp = torch.mul(x, self._norm_matrix)
        x = torch.mul(self._out_gain, self.layers(temp))
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
        config = vehicleDynamic()
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
        '''
        params: obs: [u, v, w, x, y, phi] + 
                     [delta_y_in_ref, phi_sub_phi_ref, u_sub_u_ref, u_ref] + 
                     [x_ref, y_ref, phi_ref, u_ref] * N
        return: x: [u, v, w, [delta_x, delta_y, delta_phi] * N]
        '''
        x = torch.zeros([1, self.inputSize])
        x[:, :3] = obs[:, :3]
        obs_ego = torch.concat([obs[:, 3:6], torch.zeros(1, 1)], dim = 1) 
        tempState = obs[:, 10: 10+4*self.refNum] - obs_ego.repeat(1, self.refNum)
        # calculate the first relative state
        # TODO: check the calculation of relative state
        phi_ref = obs[:, 5] - obs[:, 7]
        x_sub_x_ref = -torch.sin(phi_ref*np.pi/180) * obs[:, 6]
        y_sub_y_ref = torch.cos(phi_ref*np.pi/180) * obs[:, 6]

        x[:, 3] = (-x_sub_x_ref) * torch.cos(obs[:, 5]*np.pi/180) + (-y_sub_y_ref) * torch.sin(obs[:, 5]*np.pi/180)
        x[:, 4] = (-x_sub_x_ref) * (-torch.sin(obs[:, 5])*np.pi/180) + (-y_sub_y_ref) * torch.cos(obs[:, 5]*np.pi/180)
        x[:, 5] = torch.cos(-obs[:, 7]*np.pi/180)
        x[:, 6] = torch.sin(-obs[:, 7]*np.pi/180)

        for i in range(1, self.refNum):
            relIndex = 4 * i + 3
            tempIndex = 4 * i - 4
            x[:, relIndex] = tempState[:, tempIndex] * torch.cos(obs[:, 5]*np.pi/180) + tempState[:, tempIndex+1] * torch.sin(obs[:, 5]*np.pi/180)
            x[:, relIndex + 1] = tempState[:, tempIndex] * (-torch.sin(obs[:, 5]*np.pi/180)) + tempState[:, tempIndex+1] *  torch.cos(obs[:, 5]*np.pi/180)
            x[:, relIndex + 2] = torch.cos(tempState[:, tempIndex + 2]*np.pi/180)
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
