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
from config import vehicleDynamic
import os

class TrackingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        config = vehicleDynamic()
        # 参考速度
        self.refV = config.refV
        self.curveK = config.curveK
        self.curveA = config.curveA
        # 固定参考点向前看个数
        self.refStep = config.refStep
        # 车辆参数
        self.T = config.T  # 时间间隔
        self.m = config.m  # 自车质量
        self.a = config.a  # 质心到前轴的距离
        self.b = config.b  # 质心到后轴的距离
        self.kf = config.kf  # 前轮总侧偏刚度
        self.kr = config.kr  # 后轮总侧偏刚度
        self.Iz = config.Iz  # 转动惯量

        self.initState = config.initState
        self.testStep = config.testStep
        self.renderStep = config.renderStep

        # TODO: 范围
        # 动作空间 u = [acc, delta]
        # 数值和Actor输出相关联
        self.actionLow = [-4, -np.pi/9]
        self.actionHigh = [4, np.pi/9]
        self.actionSpace = \
            spaces.Box(low=np.array(self.actionLow),
                       high=np.array(self.actionHigh), dtype=np.float32)

        # 状态空间 x = [x, y, phi, u, v, omega, xr, yr]
        self.stateLow = [-inf, -2*self.curveA, -np.pi, 0, -5*self.refV, -20]
        self.stateHigh = [inf, 2*self.curveA, np.pi, 5*self.refV, 5*self.refV, 20]
        self.stateDim = 8

    def reset(self, batchSize):
        # 状态空间 x = [x, y, phi, u, v, omega, xr, yr]
        batchState = torch.empty([batchSize, self.stateDim])
        batchState[:, 0] = torch.linspace(0, 2*np.pi/self.curveK, batchSize)  # x
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
        reward = self.calReward(state, control)  # 使用当前状态计算
        done = self.isDone(newState, control)  # 考虑一下用state还是newState
        return newState, reward, done

    def stepFix(self, state, control):
        batchSize = state.size(0)
        newState = torch.empty([batchSize, self.stateDim])
        newState[:, 0:6] = \
            torch.stack(self.vehicleDynamic(state[:, 0], state[:, 1], state[:, 2], state[:, 3],
                                            state[:, 4], state[:, 5], control[:, 0], control[:, 1]), -1)
        newState[:, 6:] = state[:, 6:]  # 参考点不变
        reward = self.calReward(state, control)  # 使用当前状态计算
        done = self.isDone(newState, control)  # 考虑一下用state还是newState
        return newState, reward, done

    def calReward(self, state, control, MPCflag = 0):
        # TODO: 设计reward，考虑变成相反数
        if MPCflag == 0 :
            reward = \
                torch.pow(state[:, 0] - state[:, 6], 2) +\
                4 * torch.pow(state[:, 1] - state[:, 7], 2) +\
                0.05 * torch.pow(control[:, 0], 2) +\
                0.01 * torch.pow(control[:, 1], 2)
        else:
            reward = \
                pow(state[0] - state[6], 2) +\
                4 * pow(state[1] - state[7], 2) +\
                0.05 * pow(control[0], 2) +\
                0.01 * pow(control[1], 2)
        return reward

    def isDone(self, state, control):
        # TODO: 偏移状态
        batchSize = state.size(0)
        done = torch.tensor([False for i in range(batchSize)])
        done[(torch.pow(state[:, 0]-state[:, 6], 2) + torch.pow(state[:, 1]-state[:, 7], 2) > 9)] = True
        done[(torch.abs(state[:, 2]) > np.pi/3 )] = True
        done[(state[:, 0] - state[:, 6] > 0 )] = True
        return done

    def vehicleDynamic(self, x_0, y_0, phi_0, u_0, v_0, omega_0, acc, delta, MPCflag = 0):
        if MPCflag == 0:
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
        else:
            x_1 = x_0 + self.T * (u_0 * cos(phi_0) - v_0 * sin(phi_0))
            y_1 = y_0 + self.T * (v_0 * cos(phi_0) + u_0 * sin(phi_0))
            phi_1 = phi_0 + self.T * omega_0
            u_1 = u_0 + self.T * acc
            v_1 = (-(self.a * self.kf - self.b * self.kr) * omega_0 + self.kf * delta * u_0 +
                self.m * omega_0 * u_0 * u_0 - self.m * u_0 * v_0 / self.T) \
                / (self.kf + self.kr - self.m * u_0 / self.T)
            omega_1 = (-self.Iz * omega_0 * u_0 / self.T - (self.a * self.kf - self.b * self.kr) * v_0
                    + self.a * self.kf * delta * u_0) \
                / ((self.a * self.a * self.kf + self.b * self.b * self.kr) - self.Iz * u_0 / self.T)
        return x_1, y_1, phi_1, u_1, v_1, omega_1

    def referencePoint(self, x, MPCflag = 0):
        # # 1. 固定参考点
        # if MPCflag == 0:
        #     n = torch.floor(x/(self.T * self.refV))
        # else:
        #     n = math.floor(x/(self.T * self.refV))
        # refx = (n + self.refStep) * (self.T * self.refV)
        # return refx, self.referenceCurve(refx, MPCflag)

        # # 2. 定长移动self.refStep步参考点
        return x + self.refStep * self.T * self.refV, self.referenceCurve(x + self.refStep * self.T * self.refV, MPCflag)

        # 3. 定长移动1步参考点
        # return x + self.T * self.refV, self.referenceCurve(x + self.T * self.refV, MPCflag)

    def referenceCurve(self, x, MPCflag = 0):
        # return torch.sqrt(x/(30*np.pi))
        if MPCflag == 0:
            return self.curveA * torch.sin(self.curveK * x)
        else:
            return self.curveA * sin(self.curveK * x)

    def calRefState(self, state):
        batchSize = state.size(0)
        refState = torch.empty([batchSize, self.stateDim - 2])
        refState[:, 0:2] = state[:, 6:] - state[:, 0:2]
        refState[:, 2:] = state[:, 2:6]
        return refState

    def policyTest(self, policy, iteration, log_dir):
        plt.figure(iteration)
        state = torch.empty([1, self.stateDim])
        state[:, :6] = torch.tensor(self.initState)
        state[:, 6:] = torch.stack(self.referencePoint(state[:, 0]), -1)
        count = 0
        x = torch.linspace(0, 30*np.pi, 1000)
        y = self.referenceCurve(x)
        plt.xlim(-5, 100)
        plt.ylim(-1.1, 1.1)
        plt.plot(x, y, color='gray')
        plt.scatter(state[:, 0], state[:, 1], color='red', s=2)
        plt.scatter(state[:, 6], state[:, 7], color='blue', s=2)
        stateADP = np.empty(0)
        controlADP = np.empty(0)
        while(count < self.testStep):
            refState = self.calRefState(state)
            control = policy(refState).detach()
            stateADP = np.append(stateADP, state[0].numpy())
            controlADP = np.append(controlADP, control[0].numpy())
            state, reward, done = self.step(state, control)
            plt.scatter(state[:, 0], state[:, 1], color='red', s=2)
            plt.scatter(state[:, 6], state[:, 7], color='blue', s=2)
            count += 1
        plt.title('iteration:'+str(iteration))
        plt.savefig(log_dir + '/iteration'+str(iteration)+'.png')
        plt.close()
        stateADP = np.reshape(stateADP, (-1, self.stateDim))
        np.savetxt(os.path.join(log_dir, "stateFullADP"+str(iteration)+".csv"), stateADP, delimiter=',')
        controlADP = np.reshape(stateADP, (-1, 2))
        np.savetxt(os.path.join(log_dir, "controlFullADP"+str(iteration)+".csv"), controlADP, delimiter=',')

    def policyRender(self, policy):
        # 初始化
        state = torch.empty([1, self.stateDim])
        state[:, :6] = torch.tensor(self.initState)
        state[:, 6:] = torch.stack(self.referencePoint(state[:, 0]), -1)

        count = 0
        plt.ion()
        plt.figure()
        x = torch.linspace(0, 30*np.pi, 1000)
        y = self.referenceCurve(x)
        plt.xlim(-5, 100)
        plt.ylim(-1.1, 1.1)
        plt.plot(x, y, color='gray')
        while(count < self.renderStep):
            refState = self.calRefState(state)
            control = policy(refState).detach()
            state, reward, done = self.step(state, control)
            plt.scatter(state[:, 0], state[:, 1], color='red', s=5)
            plt.scatter(state[:, 6], state[:, 7], color='blue', s=5)
            plt.title('x='+str(round(state[:, 0].item(), 1)) +
                      ',y='+str(round(state[:, 1].item(), 1)) +
                      ',reward='+str(round(reward.item(), 2)))
            plt.pause(0.1)
            count += 1
        plt.ioff()
        
    def initializeState(self, stateNum):
        initialState = torch.empty([stateNum, self.stateDim])
        initialState[:, 0] = torch.linspace(0, 2*np.pi/self.curveK, stateNum)  # x
        initialState[:, 1] = self.referenceCurve(initialState[:, 0])  # y
        initialState[:, 2] = torch.atan(self.curveA * self.curveK * torch.cos(self.curveK * initialState[:, 0]))
        initialState[:, 3] = torch.ones((stateNum, )) * self.refV /2 # u
        initialState[:, 4] = torch.zeros((stateNum, ))  # v
        initialState[:, 5] = torch.zeros((stateNum, ))  # omega
        initialState[:, 6:] = torch.stack(self.referencePoint(initialState[:, 0]), -1)
        return initialState

if __name__ == '__main__':
    env = TrackingEnv()
    env.initializeState(2)