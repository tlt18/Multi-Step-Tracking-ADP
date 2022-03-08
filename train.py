import numpy as np
import torch
from config import trainConfig


class Train():
    def __init__(self, env):
        self.env = env
        self.lossIteraValue = np.empty([0, 1])
        self.lossIteraPolicy = np.empty([0, 1])
        self.lossValue = []
        self.lossPolicy = []
        self.state = None
        config = trainConfig()
        self.stepForwardPEV = config.stepForwardPEV
        self.stepForwardPIM = config.stepForwardPIM
        self.batchSize = config.batchSize
        self.batchData = torch.empty([self.batchSize, self.env.stateDim])
        self.batchDataLife = torch.zeros(self.batchSize)
        self.reset()

    def reset(self):
        self.batchData = self.env.reset(self.batchSize)

    def update(self, policy):
        refState = self.env.calRefState(self.batchData)
        control = policy(refState).detach()
        self.batchData, _, done = self.env.stepFix(self.batchData, control)
        self.batchDataLife += 1
        if sum(done==1) >0 :
            self.batchData[done==1] = self.env.reset(sum(done==1))
            self.batchDataLife[done==1] = 0
        if max(self.batchDataLife) > 40:
            self.batchData[self.batchDataLife>40] =\
                 self.env.reset(sum(self.batchDataLife>40))
            self.batchDataLife[self.batchDataLife > 40] = 0

    def policyEvaluate(self, policy, value):
        refState = self.env.calRefState(self.batchData)
        valuePredict = value(refState)
        valueTaeget = torch.zeros(self.batchSize)
        with torch.no_grad():
            stateNext = self.batchData.clone()
            for _ in range(self.stepForwardPEV):
                refState = self.env.calRefState(stateNext)
                control = policy.forward(refState)
                stateNext, reward, done = self.env.stepFix(stateNext, control)
                valueTaeget += reward
            refState = self.env.calRefState(stateNext)
            valueTaeget += (~done) * value(refState)
        lossValue = torch.pow(valuePredict - valueTaeget, 2).mean() +\
            10 * torch.pow(value(value._zero_state),2)
        # lossValue = torch.pow(valuePredict - valueTaeget, 2).mean()
        value.zero_grad()
        lossValue.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
        value.opt.step()
        value.scheduler.step()
        self.lossIteraValue = np.append(
            self.lossIteraValue, lossValue.detach().numpy())

    def policyImprove(self, policy, value):
        for p in value.parameters():
            p.requires_grad = False
        stateNext = self.batchData.clone()
        valueTarget = torch.zeros(self.batchSize)
        for i in range(self.stepForwardPIM):
            refState = self.env.calRefState(stateNext)
            control = policy.forward(refState)
            stateNext, reward, done = self.env.stepFix(stateNext, control)
            valueTarget += reward
        refState = self.env.calRefState(stateNext)
        valueTarget += (~done) * value(refState)
        for p in value.parameters():
            p.requires_grad = True
        policy.zero_grad()
        lossPolicy = valueTarget.mean()
        lossPolicy.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        policy.opt.step()
        policy.scheduler.step()
        self.lossIteraPolicy = np.append(
            self.lossIteraPolicy, lossPolicy.detach().numpy())

    def calLoss(self):
        self.lossValue.append(self.lossIteraValue.mean())
        self.lossPolicy.append(self.lossIteraPolicy.mean())
        self.lossIteraValue = np.empty([0, 1])
        self.lossIteraPolicy = np.empty([0, 1])

    # def calRefState(self, state):
    #     refState = torch.empty([self.batchSize, self.env.stateDim - 2])
    #     refState[:, 0:2] = state[:, 6:] - state[:, 0:2]
    #     refState[:, 2:] = state[:, 2:6]
    #     return refState

    # def calRefState(self, state):
    #     refState = torch.empty([self.batchSize, self.env.stateDim - 2])
    #     refState[:, 0:2] = self.batchData[:, 6:] - self.batchData[:, 0:2]
    #     refState[:, 2:] = self.batchData[:, 2:6]
    #     return refState
