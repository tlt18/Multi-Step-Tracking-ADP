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
        self.renderStep = config.renderStep

        # action space
        # u = [acc, delta]
        # If you modify the range, you must modify the output range of Actor.
        self.actionLow = [-2, -0.15]
        self.actionHigh = [2, 0.15]
        self.actionSpace = \
            spaces.Box(low=np.array(self.actionLow),
                       high=np.array(self.actionHigh), dtype=np.float64)

        # state space
        # x = [u, v, omega, x, y, phi]
        # u and v are the longitudinal and lateral velocities respectively
        self.stateLow = [0, -5*self.refV, -20, -inf, -inf, -np.pi]
        self.stateHigh = [5*self.refV, 5*self.refV, 20, inf, inf, np.pi]
        self.refNum = config.refNum
        self.stateDim = 6 + 3 * self.refNum # augmented state dimensions, \bar x = [u, v, omega, [xr, yr, phir], x, y, phi]
        self.relstateDim = 3 + 4 * self.refNum # relative state, input of NN, x_r = [u, v, omega, [xe, ye, cos(phie), sin(phie)]]
    

    def seed(self, s):
        # random seed
        np.random.seed(s)
        torch.manual_seed(s)


    def resetRandom(self, stateNum, noise = 1, MPCflag = 0):
        # augmented state space \bar x = [u, v, omega, [xr, yr, phir], x, y, phi]]
        newState = torch.empty([stateNum, self.stateDim])

        # u: [self.refV/2, self.refV*3/2]
        newState[:, 0] = self.refV + (torch.rand(stateNum) - 1/2 ) * self.refV * noise
        # newState[:, 0] = torch.normal(self.refV, self.refV/4, (stateNum, ))
        
        # v: [-self.refV/8, self.refV/8]
        newState[:, 1] = (torch.rand(stateNum) - 1/2) * self.refV / 4 * noise
        # newState[:, 1] = torch.normal(0, self.refV/16, (stateNum, ))

        # omega: [-1, 1]
        newState[:, 2] = (torch.rand(stateNum) - 1/2) * 2 * noise
        # newState[:, 2] = torch.normal(0, 0.25, (stateNum, ))

        # x, y, phi
        newState[:, -3:-1] = torch.zeros((stateNum,2))
        newState[:, -1] = torch.ones(stateNum) * np.pi/4

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

            # +- self.refV * self.T / 2
            refState[:, 0] = state[:, 0] + (torch.rand(state.size(0)) - 1/2) * self.refV * self.T * noise
            refState[:, 1] = state[:, 1] + (torch.rand(state.size(0)) - 1/2) * self.refV * self.T * noise
            # refState[:, 0] = torch.normal(state[:, 0], self.refV * self.T / 4 * noise)
            # refState[:, 1] = torch.normal(state[:, 1], self.refV * self.T / 4 * noise)

            # +-0.1
            refState[:, 2] = state[:, 2] + (torch.rand(state.size(0)) - 1/2) * 0.1 * 2 * noise
            # refState[:, 2] = torch.normal(state[:, 2], 0.05 / 2 * noise)

            for i in range(1, self.refNum):
                # index of [x, y, phi]: 3 * i, 3 * i + 1, 3 * i + 2
                randL = self.refV * self.T + (torch.rand(state.size(0)) - 1/2 ) * self.refV * self.T * noise
                deltaphi = (torch.rand(state.size(0)) - 1/2) * 0.1 * 2 * noise
                refState[:, 3 * i + 2] = refState[:, 3 * i - 1] + deltaphi
                refphi = refState[:, 3 * i - 1] + deltaphi * torch.rand(state.size(0))
                refState[:, 3 * i] = refState[:, 3 * i - 3] + torch.cos(refphi) * randL
                refState[:, 3 * i + 1] = refState[:, 3 * i - 2] + torch.sin(refphi) * randL
        else:
            return self.referenceFind(torch.tensor([state]), noise=noise, MPCflag=0)[0].tolist()
        return refState


    def resetSpecificCurve(self, stateNum, curveType = 'sine', noise = 0):
        newState = torch.empty([stateNum, self.stateDim])
        newState[:, 0] = torch.ones(stateNum) * self.refV # u
        newState[:, 1] = torch.zeros(stateNum) # v
        newState[:, 2] = torch.zeros(stateNum) # omega
        if curveType == 'sine':
            newState[:, -3] = torch.rand(stateNum) *  2*np.pi/self.curveK # x
        else :
            newState[:, -3] = torch.zeros(stateNum)
        newState[:, -2:] = torch.stack(self.referenceCurve(newState[:, -3], curveType = curveType), -1)

        if curveType == 'random':
            newState[:, 3:-3] = self.referenceFind(newState[:, -3:])
        else:
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
        return newState

    def stepReal(self, state, control, curveType = 'sine', noise = 0):
        # You must initialize all state for specific curce!
        # step in real time
        newState = torch.empty_like(state)
        temp = \
            torch.stack(self.vehicleDynamic(state[:, -3], state[:, -2], state[:, -1], state[:, 0],
                                            state[:, 1], state[:, 2], control[:, 0], control[:, 1]), -1)
        newState[:, -3:] = temp[:, :3] # x, y, phi
        newState[:, :3] = temp[:, 3:] # u, v, omega
        # TODO: you can add some specific trajectory here
        newState[:, 3:-3] = self.refDynamicReal(state[:, 3:-3], curveType = curveType, noise = noise)
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
                4 * torch.pow(state[:, -3] - state[:, 3], 2) +\
                4 * torch.pow(state[:, -2] - state[:, 4], 2) +\
                10 * torch.pow(state[:, -1] - state[:, 5], 2) +\
                torch.pow(control[:, 0], 2) +\
                0.2 * torch.pow(control[:, 1], 2)
        else:
            reward = \
                4 * pow(state[-3] - state[3], 2) +\
                4 * pow(state[-2] - state[4], 2) +\
                10 * pow(state[-1] - state[5], 2) +\
                pow(control[0], 2) +\
                0.2 * pow(control[1], 2)
        return reward


    def isDone(self, state, control):
        # TODO: design condition of done
        batchSize = state.size(0)
        done = torch.tensor([False for i in range(batchSize)])
        done[(torch.pow(state[:, -3]-state[:, 3], 2) + torch.pow(state[:, -2]-state[:, 4], 2) > 4)] = True
        done[(torch.abs(state[:, -1] - state[:, 5]) > 0.3 )] = True
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
            # refDeltax = torch.sqrt(torch.pow(refState[:, -5]-refState[:, -2],2)+torch.pow(refState[:, -6]-refState[:, -3],2))
            refDeltax = self.T * self.refV
            newRefState[:, -3] = refState[:, -3] + refDeltax * torch.cos(refState[:, -1])
            newRefState[:, -2] = refState[:, -2] + refDeltax * torch.sin(refState[:, -1])
            newRefState[:, -1] = refState[:, -1]
        else:
            return self.refDynamicVirtual(torch.tensor([refState]), MPCflag = 0)[0].tolist()
        return newRefState


    def refDynamicReal(self, refState, MPCflag = 0, curveType = 'sine', noise = 0):
        maxSection = 5
        if MPCflag == 0:
            newRefState = torch.empty_like(refState) # [[xr, yr, phir]]
            newRefState[:, :-3] = refState[:, 3:]
            # TODO: add noise to refDeltax
            # refDeltax = torch.sqrt(torch.pow(refState[:, -5]-refState[:, -2],2)+torch.pow(refState[:, -6]-refState[:, -3],2))
            refDeltax = self.T * self.refV
            if curveType == 'random':
                # deltaphi = torch.rand(refState.size(0)) * 0.05 * noise
                deltaphi = 0.1 * noise
                newRefState[:, -1] = refState[:, -1] + deltaphi
                refphi = refState[:, -1] + deltaphi * torch.rand(refState.size(0)) * 0
                newRefState[:, -3] = refState[:, -3] + refDeltax * torch.cos(refphi)
                newRefState[:, -2] = refState[:, -2] + refDeltax * torch.sin(refphi)
            else:
                refNextx = refState[:, -3].clone()
                refNexty, refNextphi = self.referenceCurve(refNextx, MPCflag, curveType = curveType)
                for _ in range(maxSection):
                    refNextx = refNextx + refDeltax / maxSection * torch.cos(refNextphi)
                    refNexty, refNextphi = self.referenceCurve(refNextx, MPCflag, curveType = curveType)
                newRefState[:, -3], newRefState[:, -2], newRefState[:, -1] = refNextx, refNexty, refNextphi
        else:
            return self.refDynamicReal(torch.tensor([refState]), MPCflag = 0, curveType = curveType, noise = noise)[0].tolist()
        return newRefState
        

    def referenceCurve(self, x, MPCflag = 0,  curveType = 'sine'):
        if MPCflag == 0:
            if curveType == 'sine':
                return self.curveA * torch.sin(self.curveK * x), torch.atan(self.curveA * self.curveK * torch.cos(self.curveK * x))
            elif curveType == 'log':
                return self.curveA * torch.log(self.curveK * x + 1), torch.atan(self.curveA / ( x + 1 / self.curveK))
            elif curveType == 'random':
                return torch.zeros_like(x), torch.zeros_like(x)

        else:
            if curveType == 'sine':
                return self.curveA * sin(self.curveK * x), atan(self.curveA * self.curveK * cos(self.curveK * x))
            elif curveType == 'log':
                return self.curveA * log(self.curveK * x + 1), atan(self.curveA / ( x + 1 / self.curveK))
            elif curveType == 'random':
                return 0, 0


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


    def policyTestReal(self, policy, iteration, log_dir, curveType = 'sine', noise = 0):
        state  = self.resetSpecificCurve(1, curveType = curveType)
        count = 0
        stateADP = np.empty(0)
        controlADP = np.empty(0)
        rewardSum = 0

        reversalList = [40, self.testStepReal-40]

        while(count < self.testStepReal):
            refState = self.relStateCal(state)
            control = policy(refState).detach()
            stateADP = np.append(stateADP, state[0].numpy())
            controlADP = np.append(controlADP, control[0].numpy())

            # if count < reversalList[0] or count > reversalList[1]:
            #     state, reward, done = self.stepReal(state, control, curveType=curveType, noise = 0)
            # else:
            state, reward, done = self.stepReal(state, control, curveType=curveType, noise = noise)

            rewardSum += min(reward.item(), 100000/self.testStepReal)
            count += 1
        stateADP = np.reshape(stateADP, (-1, self.stateDim))
        controlADP = np.reshape(controlADP, (-1, 2))
        saveADP = np.concatenate((stateADP[:, -3:], stateADP[:, :3], stateADP[:, 3:-3], controlADP), 1) # [x, y, phi, u, v, omega, [xr, yr, phir], a, delta]
        with open(log_dir + "/Real_state"+str(iteration)+".csv", 'wb') as f:
            np.savetxt(f, saveADP, delimiter=',', fmt='%.4f', comments='', header="x,y,phi,u,v,omega," + "xr,yr,phir,"*self.refNum + "a,delta")
        plt.figure()
        plt.scatter(stateADP[:, -3], stateADP[:, -2], color='red', s=0.5)
        plt.scatter(stateADP[:, 3], stateADP[:, 4], color='gray', s=0.5)
        # plt.scatter(stateADP[:, -3], stateADP[:, -2],  s=20, c='red', marker='*')
        # plt.scatter(stateADP[:, 3], stateADP[:, 4], c='gray', s = 20, marker='+')
        plt.legend(labels = ['ADP', 'reference'])
        plt.axis('equal')
        plt.title('iteration:'+str(iteration))
        plt.savefig(log_dir + '/Real_iteration'+str(iteration)+'.png')
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
            with open(log_dir + "/Virtual_state"+str(iteration)+".csv", 'wb') as f:
                np.savetxt(f, saveADP, delimiter=',', fmt='%.4f', comments='', header="x,y,phi,u,v,omega," + "xr,yr,phir,"*self.refNum + "a,delta")
            plt.figure()
            plt.scatter(stateADP[:, -3], stateADP[:, -2],  s=20, c='red', marker='*')
            plt.scatter(stateADP[:, 3], stateADP[:, 4], c='gray', s = 20, marker='+')
            plt.legend(labels = ['ADP', 'reference'])
            plt.axis('equal')
            plt.title('iteration:'+str(iteration))
            plt.savefig(log_dir + '/Virtual_iteration'+str(iteration)+'.png')
            plt.close()
        return rewardSum


    def plotReward(self, rewardSum, log_dir, saveIteration):
        plt.figure()
        plt.plot(range(0,len(rewardSum)*saveIteration, saveIteration),rewardSum)
        plt.xlabel('itetation')
        plt.ylabel('reward')
        plt.savefig(log_dir + '/reward.png')
        plt.close()

if __name__ == '__main__':
    ADP_dir = './Results_dir/2022-04-08-17-04-13'
    log_dir = ADP_dir + '/test'
    os.makedirs(log_dir, exist_ok=True)
    env = TrackingEnv()
    # env.seed(0)

    policy = Actor(env.relstateDim, env.actionSpace.shape[0])
    policy.loadParameters(ADP_dir)
    # env.policyRender(policy)
    noise = 0.25
    env.curveK = 1/20
    env.curveA = 4
    env.policyTestReal(policy, 0, log_dir, curveType = 'random', noise = noise)
    env.policyTestReal(policy, 4, log_dir, curveType = 'sine', noise = 0)
    
    env.policyTestVirtual(policy, 0, log_dir, noise = noise)

    # value = Critic(env.relstateDim, 1)
    # value.loadParameters(ADP_dir)
    # state = env.resetRandom(1, noise=0)
    # refState = env.relStateCal(state) + torch.tensor([[-0.2, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0, 0]])
    # print('refState is {}, value is {}'.format(refState[0].tolist(), value(refState)[0].tolist()))


