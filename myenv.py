import torch
import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot as plt
import matplotlib.patches as mpaches
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
import time
from math import *


class TrackingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 参考速度
        self.refV = 20/3.6
        self.curve = 1/10

        # 车辆参数
        # TODO: 参数是否合理？
        self.T = 0.1  # 时间间隔
        self.m = 1520  # 自车质量
        self.a = 1.19  # 质心到前轴的距离
        self.b = 1.46  # 质心到后轴的距离
        self.kf = -155495  # 前轮总侧偏刚度
        self.kr = -155495  # 后轮总侧偏刚度
        self.Iz = 2642  # 转动惯量

        # 动作空间 u = [acc, delta]
        # 数值和Actor输出相关联
        self.actionLow = torch.tensor([-4, -np.pi/9], dtype=torch.float32)
        self.actionHigh = torch.tensor([4, np.pi/9], dtype=torch.float32)
        self.actionSpace = \
            spaces.Box(low=self.actionLow.numpy(),
                       high=self.actionHigh.numpy(), dtype=np.float32)

        # 状态空间 x = [x, y, phi, u, v, omega, xr, yr]
        self.stateDim = 8

    def reset(self, batchSize):
        # 状态空间 x = [x, y, phi, u, v, omega, xr, yr]
        batchState = torch.empty([batchSize, self.stateDim])
        batchState[:, 0] = torch.linspace(0, 60*np.pi, batchSize)  # x
        refy = self.referenceCurve(batchState[:, 0])
        batchState[:, 1] = torch.normal(refy, torch.abs(refy/3))  # y
        batchState[:, 2] = torch.normal(0.0, np.pi/10, (batchSize, ))  # phi
        batchState[:, 3] = torch.normal(
            self.refV, self.refV/3, (batchSize, ))  # u
        batchState[:, 4] = torch.normal(0, self.refV/3, (batchSize, ))  # v
        batchState[:, 5] = torch.normal(0, 0.06, (batchSize, ))  # omega
        batchState[:, 6:] = torch.stack(
            self.referencePoint(batchState[:, 0]), -1)
        return batchState

    def step(self, state, control):
        batchSize = state.size(0)
        newState = torch.empty([batchSize, self.stateDim])
        newState[:, 0:6] = \
            torch.stack(self.vehicleDynamic(state[:, 0], state[:, 1], state[:, 2], state[:, 3],
                                            state[:, 4], state[:, 5], control[:, 0], control[:, 1]), -1)
        newState[:, 6:] = torch.stack(self.referencePoint(newState[:, 0]), -1)
        reward = self.calReward(state, control)  # 使用当前的状态计算
        done = self.isDone(newState, control)  # 考虑一下用state还是newState
        return newState, reward, done

    def stepFix(self, state, control):
        batchSize = state.size(0)
        newState = torch.empty([batchSize, self.stateDim])
        newState[:, 0:6] = \
            torch.stack(self.vehicleDynamic(state[:, 0], state[:, 1], state[:, 2], state[:, 3],
                                            state[:, 4], state[:, 5], control[:, 0], control[:, 1]), -1)
        newState[:, 6:] = state[:, 6:]  # 参考点不变
        reward = self.calReward(state, control)  # 使用当前的状态计算
        done = self.isDone(newState, control)  # 考虑一下用state还是newState
        return newState, reward, done

    def calReward(self, state, control):
        # TODO: 设计reward
        reward = \
            torch.pow(state[:, 0] - state[:, 6], 2) +\
            2 * torch.pow(state[:, 1] - state[:, 7], 2) +\
            0.05 * torch.pow(control[:, 0], 2) +\
            0.01 * torch.pow(control[:, 1], 2)
        return reward

    def isDone(self, state, control):
        # TODO: 偏移状态
        batchSize = state.size(0)
        done = torch.zeros(batchSize)
        done[torch.pow(state[:, 0]-state[:, 6], 2)+torch.pow(state[:, 1]-state[:, 7], 2) > 9] = 1
        return done

    def vehicleDynamic(self, x_0, y_0, phi_0, u_0, v_0, omega_0, acc, delta):
        x_1 = x_0 + self.T * (u_0 * torch.cos(phi_0) - v_0 * torch.sin(phi_0))
        y_1 = y_0 + self.T * (v_0 * torch.cos(phi_0) + u_0 * torch.sin(phi_0))
        phi_1 = phi_0 + self.T * omega_0
        u_1 = u_0 + self.T * acc
        v_1 = (-(self.a * self.kf - self.b * self.kr) * omega_0 + self.kf * delta * u_0 +
               self.m * omega_0 * u_0 * u_0 - self.m * u_0 * v_0 / self.T) \
            / (self.kf + self.kr - self.m * u_0 / self.T)
        omega_1 = (-self.Iz * omega_0 * u_0 / self.T - (self.a * self.kf - self.b * self.kr) * v_0
                   + self.a * self.kf * delta * u_0) \
            / ((self.a * self.a * self.kf + self.b * self.b * self.kr) - self.Iz * u_0 / self.T)
        return x_1, y_1, phi_1, u_1, v_1, omega_1

    def referencePoint(self, x):
        return x + self.T * self.refV, torch.sin(self.curve * (x + self.T * self.refV))

    def referenceCurve(self, x):
        return torch.sin(self.curve * x)

    def calRefState(self, state):
        batchSize = state.size(0)
        refState = torch.empty([batchSize, self.stateDim - 2])
        refState[:, 0:2] = state[:, 6:] - state[:, 0:2]
        refState[:, 2:] = state[:, 2:6]
        return refState

    def policyTest(self, policy, iteration, log_dir):
        plt.figure(iteration)
        state = torch.empty([1, self.stateDim])
        state[:, 0] = 0  # x
        state[:, 1] = 0  # y
        state[:, 2] = 0.03332  # phi
        state[:, 3] = self.refV  # u
        state[:, 4] = 0  # v
        state[:, 5] = 0  # omega
        state[:, 6:] = torch.stack(self.referencePoint(state[:, 0]), -1)
        count = 0
        x = torch.linspace(1,30*np.pi,1000)
        y = self.referenceCurve(x)
        plt.xlim(-5, 100)
        plt.ylim(-1.1, 1.1)
        plt.plot(x, y, color = 'gray')
        while(count < 200):
            refState = self.calRefState(state)
            control = policy(refState).detach()
            state, reward, done = self.step(state, control)
            plt.scatter(state[:, 0], state[:, 1], color = 'red', s = 5)
            plt.scatter(state[:, 6], state[:, 7], color = 'blue', s = 5)
            count += 1
        plt.title('iteration:'+str(iteration))
        plt.savefig(log_dir + '/iteration'+str(iteration)+'.png')
        plt.close()

    def policyRender(self, policy):
        # 初始化
        state = torch.empty([1, self.stateDim])
        state[:, 0] = 0  # x
        state[:, 1] = 0  # y
        state[:, 2] = 0.03332  # phi
        state[:, 3] = self.refV  # u
        state[:, 4] = 0  # v
        state[:, 5] = 0  # omega
        state[:, 6:] = torch.stack(self.referencePoint(state[:, 0]), -1)

        count = 0
        plt.ion()
        plt.figure()
        x = torch.linspace(1,30*np.pi,1000)
        y = self.referenceCurve(x)
        plt.xlim(-5, 100)
        plt.ylim(-1.1, 1.1)
        plt.plot(x, y, color = 'gray')
        while(count < 100):
            refState = self.calRefState(state)
            control = policy(refState).detach()
            state, reward, done = self.step(state, control)
            plt.scatter(state[:, 0], state[:, 1], color = 'red', s = 5)
            plt.scatter(state[:, 6], state[:, 7], color = 'blue', s = 5)
            plt.title('x='+str(round(state[:, 0].item(),1))+\
                ',y='+str(round(state[:, 1].item(),1))+\
                ',reward='+str(round(reward.item(),2)))
            plt.pause(0.1)
            count += 1
        plt.ioff()


