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
        # reference velocity
        self.refV = config.refV
        self.curveK = config.curveK
        self.curveA = config.curveA
        self.DLCh = config.DLCh
        self.DLCa = config.DLCa
        self.DLCb = config.DLCb
        self.curvePhi = config.curvePhi
        # vehicle parameters
        self.T = config.T  # time interval
        self.m = config.m  # mass
        self.a = config.a  # the center of mass to the front axis
        self.b = config.b  #  the center of mass to the rear axis
        self.kf = config.kf  # total lateral stiffness of front wheels
        self.kr = config.kr  # total lateral stiffness of rear wheels
        self.Iz = config.Iz  # rotational inertia

        self.initState = config.initState
        self.testStepReal = config.testStepReal
        self.testStepVirtual = config.testStepVirtual
        self.testSampleNum = config.testSampleNum

        # action space
        # u = [acc, delta]
        # If you modify the range, you must modify the output range of Actor.
        self.actionLow = [-2, -0.3]
        self.actionHigh = [2, 0.3]
        self.actionSpace = \
            spaces.Box(low=np.array(self.actionLow),
                       high=np.array(self.actionHigh), dtype=np.float64)

        # state space
        # x = [u, v, omega, x, y, phi]
        # u and v are the longitudinal and lateral velocities respectively
        self.stateLow = [0, -5*self.refV, -20, -inf, -inf, -2 * np.pi]
        self.stateHigh = [5*self.refV, 5*self.refV, 20, inf, inf, 2 * np.pi]
        self.changeRefNum(config.refNum)
        self.randomTestNum = 0
        self.MPCState = None
        self.MPCAction = None

    def changeRefNum(self, refNum):
        self.refNum = refNum
        self.stateDim = 6 + 3 * self.refNum # augmented state dimensions, \bar x = [u, v, omega, [xr, yr, phir], x, y, phi]
        self.relstateDim = 3 + 4 * self.refNum # relative state, input of NN, x_r = [u, v, omega, [xe, ye, cos(phie), sin(phie)]]

    def randomTestReset(self):
        self.randomTestNum = 0

    def seed(self, s):
        # random seed
        np.random.seed(s)
        torch.manual_seed(s)


    def resetRandom(self, stateNum, noise = 1, MPCflag = 0):
        # augmented state space \bar x = [u, v, omega, [xr, yr, phir], x, y, phi]
        newState = torch.empty([stateNum, self.stateDim])
        # u: [4*self.refV/5, 6*self.refV/5]
        newState[:, 0] = self.refV + 2 * (torch.rand(stateNum) - 1/2 ) * self.refV / 5 * noise
        # v: [-self.refV/5, self.refV/5]
        newState[:, 1] = 2 * (torch.rand(stateNum) - 1/2) * self.refV / 5 * noise
        # omega: [-1, 1]
        newState[:, 2] = 2 * (torch.rand(stateNum) - 1/2) * 1 * noise
        # x, y, phi
        newState[:, -3:-1] = torch.zeros((stateNum,2))
        newState[:, -1] = torch.zeros(stateNum)
        # [xr, yr, phir]
        newState[:, 3:-3] = self.referenceFind(newState[:, -3:], noise=noise)
        if MPCflag == 0:
            return newState
        else:
            return newState[0].tolist()
         

    def referenceFind(self, state, noise = 0, MPCflag = 0):
        # input: state = [x, y, phi]
        # output: N steps reference point
        if MPCflag == 0:
            refState = torch.empty((state.size(0), 3 * self.refNum))
            # +- self.refV * self.T * 1.5
            refState[:, 0] = state[:, 0] + 2 * (torch.rand(state.size(0)) - 1/2) * self.refV * self.T * 1.5 * noise
            refState[:, 1] = state[:, 1] + 2 * (torch.rand(state.size(0)) - 1/2) * self.refV * self.T * 1.5 * noise
            # +-pi/15
            refState[:, 2] = state[:, 2] + 2 * (torch.rand(state.size(0)) - 1/2) * np.pi / 15 * noise
            # refState[:, 2] = torch.normal(state[:, 2], 0.05 / 2 * noise)
            for i in range(1, self.refNum):
                # index of [x, y, phi]: 3 * i, 3 * i + 1, 3 * i + 2
                randL = self.refV * self.T + 2 * (torch.rand(state.size(0)) - 1/2) * self.refV * self.T / 5 * noise
                # +-pi/15
                deltaphi = 2 * (torch.rand(state.size(0)) - 1/2) * np.pi / 15 * noise
                refState[:, 3 * i + 2] = refState[:, 3 * i - 1] + deltaphi
                # +-pi/15
                refphi = refState[:, 3 * i - 1] + 2 * (torch.rand(state.size(0)) - 1/2) * np.pi / 15 * noise
                refState[:, 3 * i] = refState[:, 3 * i - 3] + torch.cos(refphi) * randL
                refState[:, 3 * i + 1] = refState[:, 3 * i - 2] + torch.sin(refphi) * randL
        else:
            return self.referenceFind(torch.tensor([state]), noise=noise, MPCflag=0)[0].tolist()
        return refState


    def resetSpecificCurve(self, stateNum, curveType = 'sine'):
        # \bar x = [u, v, omega, [xr, yr, phir], x, y, phi]
        newState = torch.empty([stateNum, self.stateDim])
        newState[:, 0] = torch.ones(stateNum) * self.refV # u
        newState[:, 1] = torch.zeros(stateNum) # v
        newState[:, 2] = torch.zeros(stateNum) # omega
        if curveType == 'sine':
            newState[:, -3] = torch.rand(stateNum) *  2 * np.pi/self.curveK # x
        else :
            newState[:, -3] = torch.zeros(stateNum)
        newState[:, -2:] = torch.stack(self.referenceCurve(newState[:, -3], curveType = curveType), -1) # y, phi
        # [[xr, yr, phir]]
        if curveType == 'sine' or curveType == 'DLC':
            newState[:, 3:6] = newState[:, -3:] # input of the function
            maxSection = 5
            for i in range(1, self.refNum):
                refNextx = newState[:, 3 * i].clone()
                refNexty = newState[:, 3 * i + 1].clone()
                refNextphi = newState[:, 3 * i + 2].clone()
                for _ in range(maxSection):
                    refNextx = refNextx + self.refV * self.T / maxSection * torch.cos(refNextphi)
                    refNexty, refNextphi = self.referenceCurve(refNextx, curveType = curveType)
                newState[:, 3 * i + 3] = refNextx
                newState[:, 3 * i + 4] = refNexty
                newState[:, 3 * i + 5] = refNextphi
        elif curveType == 'TurnLeft' or 'TurnRight' or 'RandomTest':
            newState[:, 3:-3] = self.referenceFind(newState[:, -3:], noise = 0, MPCflag = 0) # zeros
            if curveType == 'RandomTest':
                self.randomTestNum = 0
                # self.randomPhi = 2 * (torch.rand((self.testStepReal['RandomTest'],1)) - 1/2) * 2
                self.randomPhi = torch.normal(torch.zeros((self.testStepReal['RandomTest'],1)), 1)
                # self.randomL = 2 * (torch.rand((self.testStepReal['RandomTest'],1)) - 1/2) * 2
                self.randomL = torch.normal(torch.zeros((self.testStepReal['RandomTest'],1)), 1)
                weight = 0.1
                for i in range(1, self.testStepReal['RandomTest']):
                    self.randomPhi[i][0] = weight * self.randomPhi[i][0] + (1-weight) * self.randomPhi[i-1][0]
                    self.randomL[i][0] = weight * self.randomL[i][0] + (1-weight) * self.randomL[i-1][0]
        if curveType != 'RandomTest':
            newState[:, -2] += 2 * (torch.rand(stateNum) - 1/2) * 0.2
        return newState

    def stepReal(self, state, control, curveType = 'sine'):
        # You must initialize all state for specific curce!
        # step in real time
        # \bar x = [u, v, omega, [xr, yr, phir], x, y, phi]
        newState = torch.empty_like(state)
        temp = \
            torch.stack(self.vehicleDynamic(state[:, -3], state[:, -2], state[:, -1], state[:, 0],
                                            state[:, 1], state[:, 2], control[:, 0], control[:, 1]), -1)
        newState[:, -3:] = temp[:, :3] # x, y, phi
        newState[:, :3] = temp[:, 3:] # u, v, omega
        # you can add reference dynamics here
        newState[:, 3:-3] = self.refDynamicReal(state[:, 3:-3], MPCflag = 0, curveType = curveType)
        reward = self.calReward(state, control)  # calculate using current state
        done = self.isDone(newState, control)
        return newState, reward, done


    def stepVirtual(self, state, control):
        newState = torch.empty_like(state)
        temp = \
            torch.stack(self.vehicleDynamic(state[:, -3], state[:, -2], state[:, -1], state[:, 0],
                                            state[:, 1], state[:, 2], control[:, 0], control[:, 1]), -1)
        newState[:, -3:] = temp[:, :3] # x, y, phi
        newState[:, :3] = temp[:, 3:] # u, v, omega
        newState[:, 3:-3] = self.refDynamicVirtual(state[:, 3:-3])
        reward = self.calReward(state, control)
        done = self.isDone(newState, control)
        return newState, reward, done


    def calReward(self, state, control, MPCflag = 0):
        # TODO: design reward
        if MPCflag == 0 :
            reward = \
                15 * torch.pow(state[:, -3] - state[:, 3], 2) +\
                15 * torch.pow(state[:, -2] - state[:, 4], 2) +\
                10 * torch.pow(state[:, -1] - state[:, 5], 2) +\
                2 * torch.pow(control[:, 0], 2) +\
                2 * torch.pow(control[:, 1], 2)
        else:
            reward = \
                15 * pow(state[-3] - state[3], 2) +\
                15 * pow(state[-2] - state[4], 2) +\
                10 * pow(state[-1] - state[5], 2) +\
                2 * pow(control[0], 2) +\
                2 * pow(control[1], 2)
        return reward


    def isDone(self, state, control):
        # TODO: design condition of done
        batchSize = state.size(0)
        done = torch.tensor([False for i in range(batchSize)])
        done[(torch.pow(state[:, -3]-state[:, 3], 2) + torch.pow(state[:, -2]-state[:, 4], 2) > 4)] = True
        done[(torch.abs(state[:, -1] - state[:, 5]) > np.pi/6)] = True
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
        return [x_1, y_1, phi_1, u_1, v_1, omega_1]


    def refDynamicVirtual(self, refState, MPCflag = 0):
        # Input: N steps ref point
        # Output: N steps ref point
        if MPCflag == 0:
            newRefState = torch.empty_like(refState)
            newRefState[:, :-3] = refState[:, 3:]
            # TODO: add noise to refDeltx
            refDeltax = torch.sqrt(torch.pow(refState[:, -5]-refState[:, -2],2) + torch.pow(refState[:, -6]-refState[:, -3],2))
            # refDeltax = self.T * self.refV
            newRefState[:, -3] = refState[:, -3] + refDeltax * torch.cos(refState[:, -1])
            newRefState[:, -2] = refState[:, -2] + refDeltax * torch.sin(refState[:, -1])
            newRefState[:, -1] = refState[:, -1]
        else:
            return self.refDynamicVirtual(torch.tensor([refState]), MPCflag = 0)[0].tolist()
        return newRefState


    def refDynamicReal(self, refState, MPCflag = 0, curveType = 'sine'):
        maxSection = 5
        if MPCflag == 0:
            newRefState = torch.empty_like(refState) # [[xr, yr, phir]]
            newRefState[:, :-3] = refState[:, 3:]
            refDeltax = self.T * self.refV
            if curveType == 'sine' or curveType == 'DLC':
                refNextx = refState[:, -3].clone()
                refNexty, refNextphi = self.referenceCurve(refNextx, MPCflag, curveType = curveType)
                for _ in range(maxSection):
                    refNextx = refNextx + refDeltax / maxSection * torch.cos(refNextphi)
                    refNexty, refNextphi = self.referenceCurve(refNextx, MPCflag, curveType = curveType)
                newRefState[:, -3], newRefState[:, -2], newRefState[:, -1] = refNextx, refNexty, refNextphi
            elif curveType == 'TurnLeft':
                newRefState[:, -1] = refState[:, -1] + self.curvePhi
                refphi = refState[:, -1]
                newRefState[:, -3] = refState[:, -3] + refDeltax * torch.cos(refphi)
                newRefState[:, -2] = refState[:, -2] + refDeltax * torch.sin(refphi)
            elif curveType == 'TurnRight':
                newRefState[:, -1] = refState[:, -1] - self.curvePhi
                refphi = refState[:, -1]
                newRefState[:, -3] = refState[:, -3] + refDeltax * torch.cos(refphi)
                newRefState[:, -2] = refState[:, -2] + refDeltax * torch.sin(refphi)
            elif curveType == 'RandomTest':        
                randomPhi = self.randomPhi[self.randomTestNum]
                randomL = self.randomL[self.randomTestNum]
                self.randomTestNum += 1
                newRefState[:, -1] = refState[:, -1] + randomPhi * self.curvePhi
                refphi = refState[:, -1]
                refDeltax = self.T * self.refV + randomL * self.refV * self.T / 5
                newRefState[:, -3] = refState[:, -3] + refDeltax * torch.cos(refphi)
                newRefState[:, -2] = refState[:, -2] + refDeltax * torch.sin(refphi)
        else:
            return self.refDynamicReal(torch.tensor([refState]), MPCflag = 0, curveType = curveType)[0].tolist()
        return newRefState
        

    def referenceCurve(self, x, MPCflag = 0,  curveType = 'sine'):
        if MPCflag == 0:
            if curveType == 'sine':
                return self.curveA * torch.sin(self.curveK * x), torch.atan(self.curveA * self.curveK * torch.cos(self.curveK * x))
            elif curveType == 'DLC':
                refy = torch.empty_like(x)
                refphi = torch.empty_like(x)
                temp = (x < self.DLCa)
                refy[temp] = 0
                refphi[temp] = 0
                temp = (x > self.DLCa) & (x < 2 * self.DLCa)
                refy[temp] = self.DLCh / self.DLCa * (x[temp] - self.DLCa)
                refphi[temp] = torch.atan(torch.tensor(self.DLCh / self.DLCa))
                temp = (x > 2 * self.DLCa) & (x < 2 * self.DLCa + self.DLCb)
                refy[temp] = self.DLCh
                refphi[temp] = 0
                temp = (x > 2 * self.DLCa + self.DLCb) & (x < 3 * self.DLCa + self.DLCb)
                refy[temp] = - self.DLCh / self.DLCa * (x[temp] - 3 * self.DLCa - self.DLCb)
                refphi[temp] = - torch.atan(torch.tensor(self.DLCh / self.DLCa))
                temp = (x > 3 * self.DLCa + self.DLCb)
                refy[temp] = 0
                refphi[temp] = 0
                return refy, refphi
            # just for initial point
            elif curveType == 'TurnLeft':
                return torch.zeros_like(x), torch.zeros_like(x)
            elif curveType == 'TurnRight':
                return torch.zeros_like(x), torch.zeros_like(x)
            elif curveType == 'RandomTest':
                return torch.zeros_like(x), torch.zeros_like(x)
        else:
            refy, refphi = self.refDynamicReal(torch.tensor([x]), MPCflag = 0, curveType = curveType)
            return refy[0].tolist(), refphi[0].tolist()


    def relStateCal(self, state):
        # state = [u, v, omega, [xr, yr, phir], x, y, phi]
        batchSize = state.size(0)
        relState = torch.empty([batchSize, self.relstateDim])
        relState[:, :3] = state[:, :3]
        tempState = state[:, 3:-3] - state[:, -3:].repeat(1, self.refNum) # difference of state isn't relative state
        for i in range(self.refNum):
            relIndex = 4 * i + 3
            tempIndex = 3 * i
            relState[:, relIndex] = tempState[:, tempIndex] * torch.cos(state[:, -1]) + tempState[:, tempIndex+1] * torch.sin(state[:, -1])
            relState[:, relIndex + 1] = tempState[:, tempIndex] * (-torch.sin(state[:, -1])) + tempState[:, tempIndex+1] *  torch.cos(state[:, -1])
            relState[:, relIndex + 2] = torch.cos(tempState[:, tempIndex + 2])
            relState[:, relIndex + 3] = torch.sin(tempState[:, tempIndex + 2])
        return relState


    def policyTestReal(self, policy, iteration, log_dir, curveType = 'sine'):
        state  = self.resetSpecificCurve(1, curveType = curveType)
        count = 0
        stateADP = np.empty(0)
        controlADP = np.empty(0)
        rewardSum = 0
        while(count < self.testStepReal[curveType]):
            refState = self.relStateCal(state)
            control = policy(refState).detach()
            stateADP = np.append(stateADP, state[0].numpy())
            controlADP = np.append(controlADP, control[0].numpy())
            state, reward, done = self.stepReal(state, control, curveType=curveType)
            rewardSum += min(reward.item(), 100000/self.testStepReal[curveType])
            count += 1
        stateADP = np.reshape(stateADP, (-1, self.stateDim))
        controlADP = np.reshape(controlADP, (-1, 2))
        saveADP = np.concatenate((stateADP[:, -3:], stateADP[:, :3], stateADP[:, 3:-3], controlADP), 1) # [x, y, phi, u, v, omega, [xr, yr, phir], a, delta]
        with open(log_dir + "/Real_last_state_"+curveType+".csv", 'wb') as f:
            np.savetxt(f, saveADP, delimiter=',', fmt='%.4f', comments='', header="x,y,phi,u,v,omega," + "xr,yr,phir,"*self.refNum + "a,delta")
        plt.figure()
        plt.scatter(stateADP[:, -3], stateADP[:, -2], color='red', s=0.5)
        plt.scatter(stateADP[:, 3], stateADP[:, 4], color='gray', s=0.5)
        plt.legend(labels = ['ADP', 'reference'])
        # plt.axis('equal')
        plt.title('iteration:'+str(iteration))
        plt.savefig(log_dir + '/Real_last_iteration_'+curveType+'.png')
        plt.close()
        return rewardSum

    def policyTestVirtual(self, policy, iteration, log_dir, noise = 0, isPlot=True):
        if isPlot == True:
            state = self.resetRandom(1, noise=noise)
        else:
            state = self.resetRandom(self.testSampleNum, noise=noise)
        count = 0
        stateADP = np.empty(0)
        controlADP = np.empty(0)
        rewardSum = 0
        while(count < self.testStepVirtual):
            refState = self.relStateCal(state)
            control = policy(refState).detach()
            stateADP = np.append(stateADP, state[0].numpy())
            controlADP = np.append(controlADP, control[0].numpy())
            state, reward, done = self.stepVirtual(state, control)
            rewardSum += torch.mean(torch.min(reward,torch.tensor(50))).item()
            count += 1
        stateADP = np.reshape(stateADP, (-1, self.stateDim))
        controlADP = np.reshape(controlADP, (-1, 2))
        saveADP = np.concatenate((stateADP[:, -3:], stateADP[:, :3], stateADP[:, 3:-3], controlADP), 1)
        if isPlot == True:
            # with open(log_dir + "/Virtual_state"+str(iteration)+".csv", 'wb') as f:
            with open(log_dir + "/Virtual_last_state.csv", 'wb') as f:
                np.savetxt(f, saveADP, delimiter=',', fmt='%.4f', comments='', header="x,y,phi,u,v,omega," + "xr,yr,phir,"*self.refNum + "a,delta")
            plt.figure()
            plt.scatter(stateADP[:, -3], stateADP[:, -2],  s=20, c='red', marker='*')
            plt.scatter(stateADP[:, 3], stateADP[:, 4], c='gray', s = 20, marker='+')
            plt.legend(labels = ['ADP', 'reference'])
            # plt.axis('equal')
            plt.title('iteration:'+str(iteration))
            # plt.savefig(log_dir + '/Virtual_iteration'+str(iteration)+'.png')
            plt.savefig(log_dir + '/Virtual_last_iteration.png')
            plt.close()
        return rewardSum

    def dynamicTest(self, log_dir, actionList, noise = 0):
        for Ts in [0.01,0.05,0.005]:
            self.T = Ts
            for action in actionList:
                # augmented state space \bar x = [u, v, omega, [xr, yr, phir], x, y, phi]
                state = self.resetRandom(1, noise=noise)
                state[0][1] = -0.1
                state[0][2] = 0.1
                count = 0
                stateADP = np.empty(0)
                controlADP = np.empty(0)
                rewardSum = 0
                control = torch.tensor([action])
                while(count < int(0.4/self.T)):
                    stateADP = np.append(stateADP, state[0].numpy())
                    controlADP = np.append(controlADP, control[0].numpy())
                    state, reward, done = self.stepVirtual(state, control)
                    rewardSum += torch.mean(torch.min(reward,torch.tensor(50))).item()
                    count += 1
                stateADP = np.reshape(stateADP, (-1, self.stateDim))
                controlADP = np.reshape(controlADP, (-1, 2))
                saveADP = np.concatenate((stateADP[:, -3:], stateADP[:, :3], stateADP[:, 3:-3], controlADP), 1)
                # with open(log_dir + "/dynamicTest_a"+str(action[0])+"_delta"+str(action[1])+".csv", 'wb') as f:
                #     np.savetxt(f, saveADP, delimiter=',', fmt='%.4f', comments='', header="x,y,phi,u,v,omega," + "xr,yr,phir,"*self.refNum + "a,delta")
                plt.figure()
                plt.scatter(stateADP[:, -3], stateADP[:, -2],  s=20, c='red', marker='*')
                # plt.scatter(stateADP[:, 3], stateADP[:, 4], c='gray', s = 20, marker='+')
                # plt.legend(labels = ['ADP', 'reference'])
                # plt.legend(labels = ['ADP'])
                plt.xlabel('X [m]')
                plt.ylabel('Y [m]')
                plt.title("Ts: "+str(Ts*1000)+" [ms]")
                plt.savefig(log_dir + "/a"+str(action[0])+"_delta"+str(action[1])+'T_s'+str(self.T)+".png", bbox_inches='tight')
                plt.close()

if __name__ == '__main__':
    # ADP_dir = './Results_dir/2022-04-09-10-12-16'
    # log_dir = ADP_dir + '/test'
    # os.makedirs(log_dir, exist_ok=True)
    # env = TrackingEnv()
    # # env.seed(0)

    # policy = Actor(env.relstateDim, env.actionSpace.shape[0])
    # policy.loadParameters(ADP_dir)
    # # env.policyRender(policy)
    # noise = 0.25
    # env.curveK = 1/20
    # env.curveA = 4
    # env.policyTestReal(policy, 0, log_dir, curveType = 'random', noise = noise)
    # # env.policyTestReal(policy, 4, log_dir, curveType = 'sine', noise = 0)
    # env.policyTestVirtual(policy, 0, log_dir, noise = 0)

    # value = Critic(env.relstateDim, 1)
    # value.loadParameters(ADP_dir)
    # state = env.resetRandom(1, noise=0)
    # refState = env.relStateCal(state) + torch.tensor([[-0.2, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]])
    # print('refState is {}, value is {}'.format(refState[0].tolist(), value(refState)[0].tolist()))

    env = TrackingEnv()
    log_dir = './Simulation_dir/dynamicTest'
    actionList = []
    for a in [0]:
        for delta in [0]:
            actionList.append([a, delta])
    env.dynamicTest(log_dir, actionList, noise = 0)
