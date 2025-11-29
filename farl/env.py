"""Tracking environment and trajectory generators used across the project."""

from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces

from .config import vehicleDynamicConfig

class TrackingEnv(gym.Env):
    """Planar vehicle tracking environment with configurable reference curves."""

    def __init__(self, config: Optional[vehicleDynamicConfig] = None) -> None:
        super().__init__()
        config = config or vehicleDynamicConfig()
        self.refV = config.refV
        self.curveK = config.curveK
        self.curveA = config.curveA
        self.DLCh = config.DLCh
        self.DLCa = config.DLCa
        self.DLCb = config.DLCb
        self.curvePhi = config.curvePhi
        self.T = config.T
        self.m = config.m
        self.a = config.a
        self.b = config.b
        self.kf = config.kf
        self.kr = config.kr
        self.Iz = config.Iz

        self.initState = config.initState
        self.testStepReal = config.testStepReal
        self.testStepVirtual = config.testStepVirtual
        self.testSampleNum = config.testSampleNum
        self.actionLow = [-2, -0.3]
        self.actionHigh = [2, 0.3]
        self.actionSpace = spaces.Box(
            low=np.array(self.actionLow), high=np.array(self.actionHigh), dtype=np.float64
        )
        self.stateLow = [0, -5 * self.refV, -20, -math.inf, -math.inf, -2 * np.pi]
        self.stateHigh = [5 * self.refV, 5 * self.refV, 20, math.inf, math.inf, 2 * np.pi]
        self.changeRefNum(config.refNum)
        self.randomTestNum = 0
        self.randomLTrain = None
        self.randomPhiTrain = None
        self.randomHeadTrain = None
        self.trajectoryList = MultiRefDynamics()

    def changeRefNum(self, refNum: int) -> None:
        """Update augmented/relative state dimensions for a new horizon length."""
        self.refNum = int(refNum)
        self.stateDim = 6 + 3 * self.refNum
        self.relstateDim = 3 + 4 * self.refNum

    def randomTestReset(self):
        """Reset the random-reference generator used by RandomTest trajectories."""
        self.randomTestNum = 0

    def seed(self, s):
        """Set numpy/torch RNG seeds for deterministic evaluation."""
        np.random.seed(s)
        torch.manual_seed(s)


    def resetRandom(self, stateNum: int, noise: float = 1.0, MPCflag: int = 0):
        """Sample random augmented states ``[u,v,ω,[xr,yr,φr],x,y,φ]``."""
        newState = torch.empty([stateNum, self.stateDim])
        newState[:, 0] = self.refV + 2 * (torch.rand(stateNum) - 0.5) * self.refV / 5 * noise
        newState[:, 1] = 2 * (torch.rand(stateNum) - 0.5) * self.refV / 5 * noise
        newState[:, 2] = 2 * (torch.rand(stateNum) - 0.5) * noise
        newState[:, -3:-1] = torch.zeros((stateNum, 2))
        newState[:, -1] = torch.zeros(stateNum)
        newState[:, 3:-3] = self.referenceFind(newState[:, -3:], noise=noise)
        if MPCflag == 0:
            return newState
        return newState[0].tolist()


    def referenceFind(self, state, noise: float = 0.0, MPCflag: int = 0):
        """Generate ``refNum`` look-ahead points for the supplied poses."""
        if MPCflag == 0:
            refState = torch.empty((state.size(0), 3 * self.refNum))
            refState[:, 0] = state[:, 0] + 2 * (torch.rand(state.size(0)) - 0.5) * self.refV * self.T * 1.5 * noise
            refState[:, 1] = state[:, 1] + 2 * (torch.rand(state.size(0)) - 0.5) * self.refV * self.T * 1.5 * noise
            refState[:, 2] = state[:, 2] + 2 * (torch.rand(state.size(0)) - 0.5) * np.pi / 15 * noise
            for i in range(1, self.refNum):
                randL = self.refV * self.T + 2 * (torch.rand(state.size(0)) - 0.5) * self.refV * self.T / 5 * noise
                deltaphi = 2 * (torch.rand(state.size(0)) - 0.5) * np.pi / 15 * noise
                refState[:, 3 * i + 2] = refState[:, 3 * i - 1] + deltaphi
                refphi = refState[:, 3 * i - 1] + 2 * (torch.rand(state.size(0)) - 0.5) * np.pi / 15 * noise
                refState[:, 3 * i] = refState[:, 3 * i - 3] + torch.cos(refphi) * randL
                refState[:, 3 * i + 1] = refState[:, 3 * i - 2] + torch.sin(refphi) * randL
        else:
            return self.referenceFind(torch.tensor([state]), noise=noise, MPCflag=0)[0].tolist()
        return refState


    def resetSpecificCurve(self, stateNum: int, curveType: str = "sine"):
        """Return states initialised along a named reference curve."""
        newState = torch.empty([stateNum, self.stateDim])
        newState[:, 0] = torch.ones(stateNum) * self.refV
        newState[:, 1] = torch.zeros(stateNum)
        newState[:, 2] = torch.zeros(stateNum)
        if curveType == 'sine':
            newState[:, -3] = torch.rand(stateNum) * 2 * np.pi / self.curveK
        else:
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
        elif curveType in {'TurnLeft', 'TurnRight', 'RandomTest'}:
            newState[:, 3:-3] = self.referenceFind(newState[:, -3:], noise = 0, MPCflag = 0)
            if curveType == 'RandomTest':
                self.randomTestNum = 0
                self.randomPhi = torch.normal(torch.zeros((self.testStepReal['RandomTest'],1)), 1)
                self.randomL = torch.normal(torch.zeros((self.testStepReal['RandomTest'],1)), 1)
                weight = 0.1
                for i in range(1, self.testStepReal['RandomTest']):
                    self.randomPhi[i][0] = weight * self.randomPhi[i][0] + (1-weight) * self.randomPhi[i-1][0]
                    self.randomL[i][0] = weight * self.randomL[i][0] + (1-weight) * self.randomL[i-1][0]
        if curveType != 'RandomTest':
            newState[:, -2] += 2 * (torch.rand(stateNum) - 1/2) * 0.2
        return newState

    def stepReal(self, state, control, curveType: str = "sine"):
        """Advance the real-world dynamics by one step."""
        newState = torch.empty_like(state)
        temp = torch.stack(
            self.vehicleDynamic(
                state[:, -3], state[:, -2], state[:, -1], state[:, 0], state[:, 1], state[:, 2], control[:, 0], control[:, 1]
            ),
            -1,
        )
        newState[:, -3:] = temp[:, :3]
        newState[:, :3] = temp[:, 3:]
        newState[:, 3:-3] = self.refDynamicReal(state[:, 3:-3], MPCflag = 0, curveType = curveType)
        reward = self.calReward(state, control)
        done = self.isDone(newState, control)
        return newState, reward, done


    def stepVirtual(self, state, control, noise: float = 0.0):
        """Advance the virtual rollouts used during training."""
        newState = torch.empty_like(state)
        temp = torch.stack(
            self.vehicleDynamic(
                state[:, -3], state[:, -2], state[:, -1], state[:, 0], state[:, 1], state[:, 2], control[:, 0], control[:, 1]
            ),
            -1,
        )
        newState[:, -3:] = temp[:, :3]
        newState[:, :3] = temp[:, 3:]
        newState[:, 3:-3] = self.refDynamicVirtual(state[:, 3:-3], noise = noise)
        reward = self.calReward(state, control)
        done = self.isDone(newState, control)
        return newState, reward, done


    def calReward(self, state, control, MPCflag: int = 0):
        """Quadratic tracking cost (works on batched tensors or lists)."""
        if MPCflag == 0:
            return (
                15 * torch.pow(state[:, -3] - state[:, 3], 2)
                + 15 * torch.pow(state[:, -2] - state[:, 4], 2)
                + 10 * torch.pow(state[:, -1] - state[:, 5], 2)
                + 2 * torch.pow(control[:, 0], 2)
                + 2 * torch.pow(control[:, 1], 2)
            )
        return (
            15 * (state[-3] - state[3]) ** 2
            + 15 * (state[-2] - state[4]) ** 2
            + 10 * (state[-1] - state[5]) ** 2
            + 2 * control[0] ** 2
            + 2 * control[1] ** 2
        )


    def isDone(self, state, control):
        """Return a boolean tensor indicating if samples exceed safety bounds."""
        batchSize = state.size(0)
        done = torch.zeros(batchSize, dtype=torch.bool)
        pos_error = torch.pow(state[:, -3]-state[:, 3], 2) + torch.pow(state[:, -2]-state[:, 4], 2)
        heading_error = torch.abs(state[:, -1] - state[:, 5])
        done[pos_error > 4] = True
        done[heading_error > np.pi/6] = True
        return done


    def vehicleDynamic(self, x_0, y_0, phi_0, u_0, v_0, omega_0, acc, delta, MPCflag = 0):
        """Continuous bicycle model discretised with sampling time ``T``."""
        trig = torch if MPCflag == 0 else math
        cos_fn = trig.cos
        sin_fn = trig.sin
        x_1 = x_0 + self.T * (u_0 * cos_fn(phi_0) - v_0 * sin_fn(phi_0))
        y_1 = y_0 + self.T * (v_0 * cos_fn(phi_0) + u_0 * sin_fn(phi_0))
        phi_1 = phi_0 + self.T * omega_0
        u_1 = u_0 + self.T * acc
        v_1 = (
            -(self.a * self.kf - self.b * self.kr) * omega_0
            + self.kf * delta * u_0
            + self.m * omega_0 * u_0 * u_0
            - self.m * u_0 * v_0 / self.T
        ) / (self.kf + self.kr - self.m * u_0 / self.T)
        omega_1 = (
            -self.Iz * omega_0 * u_0 / self.T
            - (self.a * self.kf - self.b * self.kr) * v_0
            + self.a * self.kf * delta * u_0
        ) / ((self.a * self.a * self.kf + self.b * self.b * self.kr) - self.Iz * u_0 / self.T)
        return [x_1, y_1, phi_1, u_1, v_1, omega_1]


    def checkRandomTrain(self, batchSize):
        """Maintain smooth random perturbations for virtual reference motion."""
        if self.randomLTrain == None or batchSize != self.randomLTrain.size(0):
            self.randomLTrain = 2 * (torch.rand(batchSize) - 1/2)
            self.randomPhiTrain = 2 * (torch.rand(batchSize) - 1/2)
            self.randomHeadTrain = 2 * (torch.rand(batchSize) - 1/2)
        else:
            self.randomLTrain.clip(min = -1, max = 1)
            self.randomPhiTrain.clip(min = -1, max = 1)
            self.randomHeadTrain.clip(min = -1, max = 1)


    def refDynamicVirtual(self, refState, MPCflag: int = 0, noise: float = 0.0):
        """Reference propagation used during virtual training."""
        if MPCflag == 0:
            newRefState = torch.empty_like(refState)
            newRefState[:, :-3] = refState[:, 3:]
            # random noise
            weight = 0.1
            self.checkRandomTrain(refState.size(0))
            self.randomLTrain = self.randomLTrain * (1-weight) + 2 * (torch.rand(refState.size(0)) - 1/2) * weight
            self.randomPhiTrain = self.randomPhiTrain * (1-weight) + 2 * (torch.rand(refState.size(0)) - 1/2) * weight
            self.randomHeadTrain = self.randomHeadTrain * (1-weight) + 2 * (torch.rand(refState.size(0)) - 1/2) * weight
            refDeltax = torch.sqrt(
                torch.pow(refState[:, -5]-refState[:, -2],2)
                + torch.pow(refState[:, -6]-refState[:, -3],2)
                ) + self.refV * self.T / 10 * noise * self.randomLTrain
            refPhi = refState[:, -1] + np.pi / 60 * noise * self.randomPhiTrain
            refHead = refState[:, -1] + np.pi / 60 * noise * self.randomHeadTrain
            newRefState[:, -3] = refState[:, -3] + refDeltax * torch.cos(refPhi)
            newRefState[:, -2] = refState[:, -2] + refDeltax * torch.sin(refPhi)
            newRefState[:, -1] = refHead
        else:
            return self.refDynamicVirtual(torch.tensor([refState]), MPCflag = 0, noise = noise)[0].tolist()
        return newRefState


    def refDynamicReal(self, refState, MPCflag: int = 0, curveType: str = "sine"):
        """Exact propagation of nominal references for evaluation."""
        maxSection = 5
        if MPCflag == 0:
            newRefState = torch.empty_like(refState)
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


    def referenceCurve(self, x, MPCflag: int = 0, curveType: str = "sine"):
        """Return ``(y, φ)`` for the canonical curves."""
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
        """Convert augmented states into relative coordinates for the policy."""
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


    def policyTestReal(self, policy, iteration, log_dir, curveType: str = "sine"):
        """Log one rollout on a named reference curve."""
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

    def stepSpecificRef(self, state, control, info, tanLine: bool = False):
        """Rollout along the library trajectories specified in ``info``."""
        reft = info[:, 0]
        refID = info[:, 1]
        newState = torch.empty_like(state)
        temp = \
            torch.stack(self.vehicleDynamic(state[:, -3], state[:, -2], state[:, -1], state[:, 0],
                                            state[:, 1], state[:, 2], control[:, 0], control[:, 1]), -1)
        newState[:, -3:] = temp[:, :3] # x, y, phi
        newState[:, :3] = temp[:, 3:] # u, v, omega
        if tanLine == False:
            newState[:, 3:-6] = state[:, 6:-3] # ref
            newState[:, -6:-3] = torch.stack((
                self.trajectoryList.calx(reft + self.refNum * self.T, refID),
                self.trajectoryList.caly(reft + self.refNum * self.T, refID),
                self.trajectoryList.calphi(reft + self.refNum * self.T, refID)),
                dim = -1)
        else:
            # if rollout using tanLine, the info will be disabled.
            newState[:, 3:-3] = self.refDynamicVirtual(state[:, 3:-3], noise = 0)
        nextInfo = torch.empty_like(info)
        nextInfo[:, 0] = info[:, 0] + self.T
        nextInfo[:, 1] = info[:, 1]
        reward = self.calReward(state, control)
        done = self.isDone(newState, control)
        return newState, reward, done, nextInfo


    def resetSpecific(self, stateNum, noise = 1, MPCflag = 0, refIDinit = None, seed = 0, random_offset: bool = True):
        """Reset states aligned with recorded trajectories (sine/DLC/etc.)."""
        if refIDinit != None:
            refID = torch.ones(stateNum) * refIDinit
        elif refIDinit == None:
            refID = torch.floor(torch.rand(stateNum) * 3)
            refID[refID==2] = torch.zeros(sum(refID==2))
        reft = torch.zeros(stateNum)
        # reft[refID==0] = torch.rand(sum(refID==0)) * 12 * np.pi /5
        # reft[refID==1] = torch.rand(sum(refID==1)) * 28
        if noise == -1:
            reft = reft * 0
            noise = 0

        info = torch.stack((reft, refID), dim = -1)
        # \bar x = [u, v, omega, [xr, yr, phir], x, y, phi]
        newState = torch.empty([stateNum, self.stateDim])
        # u: [4*self.refV/5, 6*self.refV/5]
        newState[:, 0] = self.refV + 2 * (torch.rand(stateNum) - 1/2 ) * self.refV / 5 * noise
        # v: [-self.refV/10, self.refV/10]
        newState[:, 1] = 2 * (torch.rand(stateNum) - 1/2) * self.refV / 10 * noise
        # omega: [-0.5, 0.5]
        newState[:, 2] = 2 * (torch.rand(stateNum) - 1/2) * 0.5 * noise
        # [xr, yr, phir] * refNum
        for i in range(self.refNum):
            newState[:, 3 * i + 3] = self.trajectoryList.calx(reft + i * self.T, refID)
            newState[:, 3 * i + 4] = self.trajectoryList.caly(reft + i * self.T, refID)
            newState[:, 3 * i + 5] = self.trajectoryList.calphi(reft + i * self.T, refID)
        if random_offset:
            random_x_list = [0.0] * 6
            random_y_list = [0.3, 0.35, 0.4, 0.45, 0.5]
            random_phi_list = [0.0] * 6
            newState[:, -3] = newState[:, 3] + random_x_list[seed%5] * self.refV * self.T * 1 * noise
            newState[:, -2] = newState[:, 4] + random_y_list[seed%5] * self.refV * self.T * 1 * noise
            newState[:, -1] = newState[:, 5] + random_phi_list[seed%6] * np.pi / 15 * noise
        else:
            newState[:, -3] = newState[:, 3]
            newState[:, -2] = newState[:, 4]
            newState[:, -1] = newState[:, 5]
        if MPCflag == 0:
            return newState, info
        else:
            return newState[0].tolist(), info[0].tolist()


class MultiRefDynamics:
    """Wrapper exposing the different reference trajectories used in FAADP."""

    def __init__(self) -> None:
        self.refTrajectory = [sineCurve(1, 1/6), DLC(30.01, 50, 3.5), Circle(30), randomCurve("./Simulation_dir", 3)]

    def calx(self, t, refID, MPCflag = 0):
        if MPCflag == 0:
            x = torch.zeros_like(t)
            for i, refTraj in enumerate(self.refTrajectory):
                x = x + (refID == i) * refTraj.calx(t)
            return x
        else:
            return self.calx(torch.tensor([t]), refID, MPCflag = 0)[0].tolist()

    def caly(self, t, refID, MPCflag = 0):
        if MPCflag == 0:
            y = torch.zeros_like(t)
            for i, refTraj in enumerate(self.refTrajectory):
                y = y + (refID == i) * refTraj.caly(t)
            return y
        else:
            return self.caly(torch.tensor([t]), refID, MPCflag = 0)[0].tolist()

    def calphi(self, t, refID, MPCflag = 0):
        if MPCflag == 0:
            phi = torch.zeros_like(t)
            for i, refTraj in enumerate(self.refTrajectory):
                phi = phi + (refID == i) * refTraj.calphi(t)
            return phi
        else:
            return self.calphi(torch.tensor([t]), refID, MPCflag = 0)[0].tolist()

class randomCurve:
    """Persisted stochastic reference trajectories used for RandomTest."""

    T = 0.1
    refV = 5
    curvePhi = np.pi/40
    trjsteps = 500
    tolerance = 1e-2

    def __init__(self, data_root: str, id: int) -> None:
        self.data_root = data_root
        self.id = id
        self.file_path = f"{data_root}/randomCurve_{id}.npy"

        if os.path.exists(self.file_path):
            self.load_data()
            print(f"randomCurve_{id} loaded")
        else:
            self.generate_data()
            self.save_data()
            print(f"randomCurve_{id} generated")

    def load_data(self):
        data = np.load(self.file_path, allow_pickle=True).item()
        self.refx = torch.tensor(data['refx'])
        self.refy = torch.tensor(data['refy'])
        self.refphi = torch.tensor(data['refphi'])

    def generate_data(self):
        self.refx = torch.zeros(self.trjsteps + 100)
        self.refy = torch.zeros(self.trjsteps + 100)
        self.refphi = torch.zeros(self.trjsteps + 100)

        randomPhi = torch.normal(torch.zeros(self.trjsteps + 100), 1)
        randomL = torch.normal(torch.zeros(self.trjsteps + 100), 1)
        weight = 0.35
        for i in range(1, self.trjsteps + 100):
            randomPhi[i] = weight * randomPhi[i] + (1 - weight) * randomPhi[i - 1]
            randomL[i] = weight * randomL[i] + (1 - weight) * randomL[i - 1]
            self.refphi[i] = self.refphi[i - 1] + randomPhi[i] * self.curvePhi
            refDeltaX = self.T * self.refV + randomL[i] * self.refV * self.T / 10
            self.refx[i] = self.refx[i - 1] + refDeltaX * torch.cos(self.refphi[i])
            self.refy[i] = self.refy[i - 1] + refDeltaX * torch.sin(self.refphi[i])

    def save_data(self):
        data = {
            'refx': self.refx.numpy(),
            'refy': self.refy.numpy(),
            'refphi': self.refphi.numpy()
        }
        np.save(self.file_path, data)

    def _time_to_index(self, t):
        if isinstance(t, torch.Tensor):
            ratio = t / self.T
            rounded = torch.round(ratio)
            if torch.any(torch.abs(ratio - rounded) > self.tolerance):
                raise ValueError("t/T is not close to an integer.")
            return rounded.to(torch.long), True
        ratio = torch.as_tensor(t, dtype=torch.float32) / self.T
        rounded = torch.round(ratio)
        if torch.any(torch.abs(ratio - rounded) > self.tolerance):
            raise ValueError("t/T is not close to an integer.")
        return rounded.to(torch.long), False

    def _format_output(self, tensor: torch.Tensor, is_torch: bool):
        if is_torch:
            return tensor
        if tensor.numel() == 1:
            return tensor.item()
        return tensor.numpy()

    def calx(self, t):
        idx, is_torch = self._time_to_index(t)
        return self._format_output(self.refx[idx], is_torch)

    def caly(self, t):
        idx, is_torch = self._time_to_index(t)
        return self._format_output(self.refy[idx], is_torch)

    def calphi(self, t):
        idx, is_torch = self._time_to_index(t)
        return self._format_output(self.refphi[idx], is_torch)

class sineCurve:
    """Fixed-speed sinusoidal reference."""

    def __init__(self, A = 1, K = 1/6) -> None:
        self.A = A
        self.K = K
        self.refV = 5

    def calx(self, t: torch.Tensor) -> torch.Tensor:
        # fixed speed
        return self.refV * t

    def caly(self, t: torch.Tensor) -> torch.Tensor:
        return self.A * torch.sin(self.K * self.refV * t)

    def calphi(self, t: torch.Tensor) -> torch.Tensor:
        return torch.atan(self.A * self.K * torch.cos(self.K * self.refV * t))

class DLC:
    """Piecewise-linear double lane-change reference."""

    def __init__(self, DLCa = 30.01, DLCb = 50, DLCh = 3.5) -> None:
        self.DLCa = DLCa
        self.DLCb = DLCb
        self.DLCh = DLCh
        self.refV = 5

    def calx(self, t: torch.Tensor) -> torch.Tensor:
        # fixed speed
        return self.refV * t

    def caly(self, t: torch.Tensor) -> torch.Tensor:
        x = self.refV * t
        refy = torch.empty_like(x)
        temp = (x < self.DLCa)
        refy[temp] = 0
        temp = (x > self.DLCa) & (x < 2 * self.DLCa)
        refy[temp] = self.DLCh / self.DLCa * (x[temp] - self.DLCa)
        temp = (x > 2 * self.DLCa) & (x < 2 * self.DLCa + self.DLCb)
        refy[temp] = self.DLCh
        temp = (x > 2 * self.DLCa + self.DLCb) & (x < 3 * self.DLCa + self.DLCb)
        refy[temp] = - self.DLCh / self.DLCa * (x[temp] - 3 * self.DLCa - self.DLCb)
        temp = (x > 3 * self.DLCa + self.DLCb)
        refy[temp] = 0
        return refy

    def calphi(self, t: torch.Tensor) -> torch.Tensor:
        x = self.refV * t
        refphi = torch.empty_like(x)
        temp = (x < self.DLCa)
        refphi[temp] = 0
        temp = (x > self.DLCa) & (x < 2 * self.DLCa)
        refphi[temp] = torch.atan(torch.tensor(self.DLCh / self.DLCa))
        temp = (x > 2 * self.DLCa) & (x < 2 * self.DLCa + self.DLCb)
        refphi[temp] = 0
        temp = (x > 2 * self.DLCa + self.DLCb) & (x < 3 * self.DLCa + self.DLCb)
        refphi[temp] = - torch.atan(torch.tensor(self.DLCh / self.DLCa))
        temp = (x > 3 * self.DLCa + self.DLCb)
        refphi[temp] = 0
        return refphi

class Circle:
    """Circular path keeping a constant speed."""

    def __init__(self, R = 30) -> None:
        self.R = R
        self.refV = 5

    def calx(self, t: torch.Tensor) -> torch.Tensor:
        return self.R * torch.cos(self.refV * t / self.R)

    def caly(self, t: torch.Tensor) -> torch.Tensor:
        return self.R * torch.sin(self.refV * t / self.R)

    def calphi(self, t: torch.Tensor) -> torch.Tensor:
        return self.refV * t / self.R + np.pi/2

if __name__ == '__main__':
    env = TrackingEnv()
    print("TrackingEnv demo: state dim =", env.stateDim)
