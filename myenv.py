import math
import os
import time
from math import *

import gym
import matplotlib.patches as mpaches
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

from config import vehicleDynamic
from network import Actor, Critic


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

        # 状态空间 x = [u, v, omega, x, y, phi] u、v分别是纵向速度和横向速度
        # 由于MPC需要在虚拟轴上求解，因此需要把y的范围写大一点
        # TODO:横摆角速度的范围
        self.stateLow = [0, -5*self.refV, -20, -inf, -5*self.curveA, -np.pi]
        self.stateHigh = [5*self.refV, 5*self.refV, 20, inf, 5*self.curveA, np.pi]
        self.refNum = config.refNum
        self.stateDim = 6 + 3 * self.refNum # 增广状态维度
        self.relstateDim = 3 + 3 * self.refNum # 相对状态维度

    def reset(self, batchSize):
        # 状态空间 x = [u, v, omega, [xr, yr, phir], x, y, phi]
        batchState = torch.empty([batchSize, self.stateDim])
        batchState[:, 0] = torch.normal(self.refV, self.refV/4, (batchSize, ))  # u 纵向
        batchState[:, 1] = torch.normal(0, self.refV/5, (batchSize, ))  # v 横向
        batchState[:, 2] = torch.normal(0, 0.06, (batchSize, ))  # omega
        batchState[:, -3] = torch.rand(batchSize) * 2*np.pi/self.curveK # x
        refy, refphi = self.referenceCurve(batchState[:, -3])
        batchState[:, -2] = torch.normal(refy, torch.abs(refy/4))  # y
        batchState[:, -1] = torch.normal(refphi, torch.abs(refphi/4))  # phi
        batchState[:, 3:-3] = torch.stack(self.referenceFind(batchState[:, -3]), -1) # 通过x生成参考点
        return batchState

    def stepReal(self, state, control):
        # 在真实时域中递推参考点，使用固定参考点的方式。
        batchSize = state.size(0)
        newState = torch.empty([batchSize, self.stateDim])
        temp = \
            torch.stack(self.vehicleDynamic(state[:, -3], state[:, -2], state[:, -1], state[:, 0],
                                            state[:, 1], state[:, 2], control[:, 0], control[:, 1]), -1)
        newState[:, -3:] = temp[:, :3]
        newState[:, :3] = temp[:, 3:]
        newState[:, 3:-3] = torch.stack(self.referenceFind(state[:, 3]), -1)
        reward = self.calReward(state, control)  # 使用当前状态计算
        done = self.isDone(newState, control)  # 考虑一下用state还是newState
        return newState, reward, done

    def stepVirtual(self, state, control):
        batchSize = state.size(0)
        newState = torch.empty([batchSize, self.stateDim])
        temp = \
            torch.stack(self.vehicleDynamic(state[:, -3], state[:, -2], state[:, -1], state[:, 0],
                                            state[:, 1], state[:, 2], control[:, 0], control[:, 1]), -1)
        newState[:, -3:] = temp[:, :3]
        newState[:, :3] = temp[:, 3:]
        newState[:, 3:-3] = torch.stack(self.refdynamicvirtual(state[:, 3:-3]), -1)
        reward = self.calReward(state, control)  # 使用当前状态计算
        done = self.isDone(newState, control)  # 考虑一下用state还是newState
        return newState, reward, done

    def calReward(self, state, control, MPCflag = 0):
        # TODO: 设计reward，考虑变成相反数
        if MPCflag == 0 :
            reward = \
                torch.pow(state[:, -3] - state[:, 3], 2) +\
                4 * torch.pow(state[:, -2] - state[:, 4], 2) +\
                0.05 * torch.pow(control[:, 0], 2) +\
                0.01 * torch.pow(control[:, 1], 2)
        else:
            reward = \
                pow(state[-3] - state[3], 2) +\
                4 * pow(state[-2] - state[4], 2) +\
                0.05 * pow(control[0], 2) +\
                0.01 * pow(control[1], 2)
        return reward

    def isDone(self, state, control):
        # TODO: 设计一下
        batchSize = state.size(0)
        done = torch.tensor([False for i in range(batchSize)])
        done[(torch.pow(state[:, -3]-state[:, 3], 2) + torch.pow(state[:, -2]-state[:, 4], 2) > 9)] = True
        done[(torch.abs(state[:, -1]) > np.pi/3 )] = True
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

    def refdynamicvirtual(self, refstate, MPCflag = 0):
        # 注意这里的输出输入都是N个参考点
        if MPCflag == 0:
            refNextx = refstate[:, 0] + self.refV * self.T * torch.cos(refstate[:, 2])
            refNexty = refstate[:, 1] + self.refV * self.T * torch.sin(refstate[:, 2])
            refNextphi = refstate[:, 2]
        return refNextx, refNexty, refNextphi

    def referenceFind(self, x, MPCflag = 0):
        # input: 初始时刻的x坐标（用于第一次生成参考点），或者参考点的x坐标（用于真实时域中参考点递推）
        # output: 后续N个参考点
        if MPCflag == 0:
            n = torch.floor(x/(self.T * self.refV)) + 1
        else:
            n = math.floor(x/(self.T * self.refV)) + 1
        refx = self.T * self.refV * n
        refy, refphi = self.referenceCurve(refx, MPCflag)
        return refx, refy, refphi

    def referenceCurve(self, x, MPCflag = 0):
        if MPCflag == 0:
            return self.curveA * torch.sin(self.curveK * x), torch.atan(self.curveA * self.curveK * torch.cos(self.curveK * x))
        else:
            return self.curveA * sin(self.curveK * x), atan(self.curveA * self.curveK * cos(self.curveK * x))

    def relStateCal(self, state):
        batchSize = state.size(0)
        relState = torch.empty([batchSize, self.relstateDim])
        relState[:, :3] = state[:, :3]
        relState[:, 3:] = state[:, 3:-3] - state[:, -3:].repeat(1, self.refNum)
        return relState

    def policyTest(self, policy, iteration, log_dir):
        plt.figure(iteration)
        state = torch.empty([1, self.stateDim])
        state[:, :3] = torch.tensor(self.initState)[3:]
        state[:, -3:] = torch.tensor(self.initState)[:3]
        state[:, 3:-3] = torch.stack(self.referenceFind(state[:, -3]), -1)
        count = 0
        x = torch.linspace(0, 30*np.pi, 1000)
        y, _ = self.referenceCurve(x)
        plt.xlim(-5, 100)
        plt.ylim(-1.1, 1.1)
        plt.plot(x, y, color='gray')
        plt.scatter(state[:, -3], state[:, -2], color='red', s=2)
        plt.scatter(state[:, 3], state[:, 4], color='blue', s=2)
        stateADP = np.empty(0)
        controlADP = np.empty(0)
        while(count < self.testStep):
            refState = self.relStateCal(state)
            control = policy(refState).detach()
            stateADP = np.append(stateADP, state[0].numpy())
            controlADP = np.append(controlADP, control[0].numpy())
            state, reward, done = self.stepReal(state, control)
            plt.scatter(state[:, -3], state[:, -2], color='red', s=2)
            plt.scatter(state[:, 3], state[:, 4], color='blue', s=2)
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
        state[:, :3] = torch.tensor(self.initState)[3:]
        state[:, -3:] = torch.tensor(self.initState)[:3]
        state[:, 3:-3] = torch.stack(self.referenceFind(state[:, -3]), -1)
        count = 0
        plt.ion()
        plt.figure()
        x = torch.linspace(0, 30*np.pi, 1000)
        y, _ = self.referenceCurve(x)
        plt.xlim(-5, 100)
        plt.ylim(-1.1, 1.1)
        plt.plot(x, y, color='gray')
        while(count < self.renderStep):
            refState = self.relStateCal(state)
            control = policy(refState).detach()
            state, reward, done = self.stepReal(state, control)
            plt.scatter(state[:, -3], state[:, -2], color='red', s=5)
            plt.scatter(state[:, 3], state[:, 4], color='blue', s=5)
            plt.title('x='+str(round(state[:, -3].item(), 1)) +
                      ',y='+str(round(state[:, -2].item(), 1)) +
                      ',reward='+str(round(reward.item(), 2)))
            plt.pause(0.1)
            count += 1
        plt.ioff()
        
    def initializeState(self, stateNum):
        initialState = torch.empty([stateNum, self.stateDim])
        initialState[:, 0] = torch.ones((stateNum, )) * self.refV /2 # u 纵向
        initialState[:, 1] = torch.zeros((stateNum, ))  # v 横向
        initialState[:, 2] = torch.zeros((stateNum, ))  # omega
        initialState[:, -3] = torch.linspace(0, 2*np.pi/self.curveK, stateNum)  # x
        # initialState[:, -3] = torch.rand(stateNum) * 2 * np.pi / self.curveK # x
        refy, refphi = self.referenceCurve(initialState[:, -3])
        initialState[:, -2] = refy  # y
        initialState[:, -1] = refphi
        initialState[:, 3:-3] = torch.stack(self.referenceFind(initialState[:, -3]), -1)
        return initialState

if __name__ == '__main__':
    ADP_dir = './Results_dir/2022-03-24-23-11-27'
    env = TrackingEnv()
    policy = Actor(env.relstateDim, env.actionSpace.shape[0])
    policy.loadParameters(ADP_dir)
    env.policyRender(policy)

