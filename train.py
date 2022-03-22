import numpy as np
import torch
from config import trainConfig
import os
import matplotlib.pyplot as plt

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
        self.updFix = config.updFix
        self.batchSize = config.batchSize
        self.batchData = torch.empty([self.batchSize, self.env.stateDim])
        self.batchDataLife = torch.zeros(self.batchSize)
        self.reset()

    def reset(self):
        self.batchData = self.env.reset(self.batchSize)

    def update(self, policy):
        refState = self.env.calRefState(self.batchData)
        control = policy(refState).detach()
        if self.updFix == True:
            self.batchData, _, done = self.env.stepVirtual(self.batchData, control)
        else:
            self.batchData, _, done = self.env.stepReal(self.batchData, control)
        self.batchDataLife += 1
        if sum(done==1) >0 :
            self.batchData[done==1] = self.env.reset(sum(done==1))
            self.batchDataLife[done==1] = 0
        lifeMax = 5
        if max(self.batchDataLife) > lifeMax:
            self.batchData[self.batchDataLife > lifeMax] =\
                 self.env.reset(sum(self.batchDataLife > lifeMax))
            self.batchDataLife[self.batchDataLife > lifeMax] = 0

    def policyEvaluate(self, policy, value):
        refState = self.env.calRefState(self.batchData)
        valuePredict = value(refState)
        valueTaeget = torch.zeros(self.batchSize)
        with torch.no_grad():
            stateNext = self.batchData.clone()
            for _ in range(self.stepForwardPEV):
                refState = self.env.calRefState(stateNext)
                control = policy.forward(refState)
                stateNext, reward, done = self.env.stepVirtual(stateNext, control)
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
            stateNext, reward, done = self.env.stepVirtual(stateNext, control)
            valueTarget += reward
        refState = self.env.calRefState(stateNext)
        # valueTarget += (~done) * value(refState)
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

    def saveDate(self, log_dir):
        np.savetxt(os.path.join(log_dir, "value_loss.csv"), self.lossValue, delimiter=',')
        np.savetxt(os.path.join(log_dir, "policy_loss.csv"), self.lossPolicy, delimiter=',')
        plt.figure()
        plt.plot(range(len(self.lossValue)), self.lossValue)
        plt.xlabel('iteration')
        plt.ylabel('Value Loss')
        plt.savefig(log_dir + '/value_loss.png')
        plt.close()
        plt.figure()
        plt.plot(range(len(self.lossPolicy)), self.lossPolicy)
        plt.xlabel('iteration')
        plt.ylabel('Policy Loss')
        plt.savefig(log_dir + '/policy_loss.png')
        plt.close()