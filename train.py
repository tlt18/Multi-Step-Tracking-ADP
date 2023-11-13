import numpy as np
import torch
from config import trainConfig
import os
import matplotlib.pyplot as plt
from replaybuffer import ReplayBuffer
from myenv import TrackingEnv

class Train():
    def __init__(self, env: TrackingEnv, log_dir):
        self.env = env
        self.lossIteraValue = None
        self.lossIteraPolicy = None
        # self.lossValue = np.empty(0)
        # self.lossPolicy = np.empty(0)
        config = trainConfig()
        self.stepForwardPEV = config.stepForwardPEV
        self.batchSize = config.batchSize
        self.sampleSize = config.sampleSize
        self.warmBuffer = config.warmBuffer
        self.gammar = config.gammar
        self.lifeMax = config.lifeMax
        self.refNoise = config.refNoise
        self.statelifeMax = torch.rand(self.sampleSize) * config.lifeMax
        self.sampleData = None
        self.sapleInfo = None
        self.tanLine = config.tanLine
        self.sampleDataLife = torch.zeros(self.sampleSize)
        self.accumulateReward = None
        self.stateForwardNext = None
        self.doneForward = None
        self.gammarForward = 1
        self.buffer = ReplayBuffer(config.capacity)
        self.reset()

    def reset(self):
        self.sampleData, self.sampleInfo = self.env.resetSpecific(self.batchSize)
        for i in range(self.sampleSize):
            self.buffer.push(torch.cat([self.sampleData[i], self.sampleInfo[i]], dim = -1))

    def update(self, policy):
        relState = self.env.relStateCal(self.sampleData)
        control = policy(relState).detach()
        self.sampleData, _, done, self.sampleInfo = self.env.stepSpecificRef(self.sampleData, control, self.sampleInfo)
        self.sampleDataLife += 1
        if sum(done==True) >0:
            self.sampleData[done==1], self.sampleInfo[done==1] = self.env.resetSpecific(sum(done==1))
            self.sampleDataLife[done==1] = 0
        if sum(self.sampleDataLife > self.statelifeMax) > 0:
            temp = (self.sampleDataLife > self.statelifeMax)
            self.sampleData[temp], self.sampleInfo[temp] =self.env.resetSpecific(sum(temp))
            self.sampleDataLife[temp] = 0
            self.statelifeMax[temp] = torch.rand(sum(temp)) * self.lifeMax
        # The sampled data is stored into replaybuffer
        for i in range(self.sampleSize):
            self.buffer.push(torch.cat([self.sampleData[i], self.sampleInfo[i]], dim = -1))

    def policyEvaluate(self, policy, value):
        while len(self.buffer) < self.warmBuffer:
            self.update(policy)
        # use replay buffer
        batchData_ = torch.stack(self.buffer.sample(self.batchSize))
        self.batchData = batchData_[:, :-2]
        self.batchInfo = batchData_[:, -2:]

        relState = self.env.relStateCal(self.batchData)
        valuePredict = value(relState)
        valueTaeget = torch.zeros(self.batchSize)
        stateNext = self.batchData.clone()
        infoNext = self.batchInfo.clone()
        self.gammarForward = 1
        for _ in range(self.stepForwardPEV):
            relState = self.env.relStateCal(stateNext)
            control = policy(relState)
            stateNext, reward, done, infoNext = self.env.stepSpecificRef(stateNext, control, infoNext, tanLine = self.tanLine)
            valueTaeget += reward * self.gammarForward * (~done)
            self.gammarForward *= self.gammar
        self.accumulateReward = valueTaeget.clone()
        self.stateForwardNext = stateNext.clone()
        self.doneForward = done
        relState = self.env.relStateCal(stateNext)
        valueTaeget += (~done) * value(relState) * self.gammarForward
        valueTaeget = valueTaeget.detach()
        lossValue = torch.pow(valuePredict - valueTaeget, 2).mean()
        value.zero_grad()
        lossValue.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
        value.opt.step()
        value.scheduler.step()
        self.lossIteraValue = lossValue.detach().numpy()

    def policyImprove(self, policy, value):
        for p in value.parameters():
            p.requires_grad = False
        relState = self.env.relStateCal(self.stateForwardNext)
        valueTarget = self.accumulateReward + (~self.doneForward) * value(relState) * self.gammarForward
        for p in value.parameters():
            p.requires_grad = True
        policy.zero_grad()
        lossPolicy = valueTarget.mean()
        lossPolicy.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        policy.opt.step()
        policy.scheduler.step()
        self.lossIteraPolicy = lossPolicy.detach().numpy()