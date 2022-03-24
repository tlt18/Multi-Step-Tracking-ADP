from config import trainConfig
from myenv import TrackingEnv
from network import Actor, Critic
from train import Train
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time
import shutil

# mode setting
isTrain = True


# parameters setting
config = trainConfig()

# random seed
np.random.seed(0)
torch.manual_seed(0)

env = TrackingEnv()
env.seed(0)
stateDim = env.stateDim
actionDim = env.actionSpace.shape[0]
policy = Actor(stateDim, actionDim, lr = config.lrPolicy)
value = Critic(stateDim, 1, lr = config.lrValue)
log_dir = "./Results_dir/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(log_dir + '/train', exist_ok=True)

# TODO: 测试效果
shutil.copy('./config.py', log_dir + '/config.py')

if isTrain:
    print("----------------------Start Training!----------------------")
    train = Train(env)
    iterarion = 0
    lossListValue = 0
    while iterarion < config.iterationMax:
        # PEV
        train.policyEvaluate(policy, value)
        # PIM
        train.policyImprove(policy, value)
        train.calLoss()
        # update
        train.update(policy)
        if iterarion % config.iterationPrint == 0:
            print("iteration: {}, LossValue: {}, LossPolicy: {}".format(
                iterarion, train.lossValue[-1], train.lossPolicy[-1]))
        if iterarion % config.iterationSave == 0 or iterarion == config.iterationMax - 1:
            env.policyTest(policy, iterarion, log_dir+'/train')
            value.saveParameters(log_dir)
            policy.saveParameters(log_dir)
            train.saveDate(log_dir+'/train')
            # env.policyRender(policy)
        iterarion += 1


