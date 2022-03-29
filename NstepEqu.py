import math
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import MPCConfig
from myenv import TrackingEnv
from network import Actor, Critic
from solver import Solver

##################
# 需要改solve参考点更新代码


def simulationOpen(MPCStep, simu_dir):
    # 虚拟时域中MPC表现（开环控制）
    env = TrackingEnv()
    solver = Solver()
    for mpcstep in MPCStep:
        print("----------------------Start Solving!----------------------")
        print("MPCStep: {}".format(mpcstep))
        plt.figure(mpcstep)
        tempstate = env.initializeState(200)
        tempstate = tempstate[0].tolist() # 这里就取了一个，可以考虑从0开始取
        state = tempstate[-3:] + tempstate[:3]
        refState = tempstate[3:6]
        # x = torch.linspace(0, 30*np.pi, 1000)
        # y, _ = env.referenceCurve(x)
        # plt.xlim(-5, 100)
        # plt.ylim(-1.1, 1.1)
        # plt.plot(x, y, color='gray')
        count = 0
        _, control = solver.MPCSolver(state, refState, mpcstep, isReal=False)
        plt.scatter(state[0], state[1], color='red', s=5)
        plt.scatter(refState[0], refState[1], color='blue', s=5)
        stateList = np.empty(0)
        refList = np.empty(0)
        stateList = np.append(stateList, state)
        refList = np.append(refList, refState)
        while(count < mpcstep):
            action = control[count].tolist()
            state = env.vehicleDynamic(
                state[0], state[1], state[2], state[3], state[4], state[5], action[0], action[1], MPCflag=1)
            refState = env.refDynamicVirtualx(refState, mpcstep, count + 1, MPCflag=1)[:3]
            stateList = np.append(stateList, state)
            refList = np.append(refList, refState)
            plt.scatter(state[0], state[1], color='red', s=5)
            plt.scatter(refState[0], refState[1], color='blue', s=5)
            count += 1
        plt.title('MPCStep:'+str(mpcstep))
        plt.savefig(simu_dir + '/simulationOpenX'+str(mpcstep)+'.png')
        plt.close()
        stateList = np.reshape(stateList, (-1, 6))
        refList = np.reshape(refList, (-1, env.refNum * 3))
        # if mpcstep==MPCStep[-1]:
        #     animationPlot(stateList[:,:2], refList[:,:2],'x', 'y')

def comparePlot(MPCAll, ADP, xCoor, MPCStep, xName, yName, simu_dir, title):
    plt.figure()
    colorList = ['green', 'darkorange', 'blue', 'yellow']
    for i in range(len(MPCAll)):
        MPC = MPCAll[i]
        plt.plot(xCoor, MPC, linewidth=2, color = colorList[i], linestyle = '-.')
    plt.plot(xCoor, ADP, linewidth = 2, color='red',linestyle = '-')
    plt.legend(labels=['MPC'+str(mpcStep) for mpcStep in MPCStep] + ['ADP'])
    plt.xlabel(xName)
    plt.ylabel(yName)
    # plt.subplots_adjust(left=)
    plt.savefig(simu_dir + '/' + title + '.png', bbox_inches='tight')
    plt.close()

def animationPlot(state, refstate, xName, yName):
    plt.figure()
    plt.ion()
    plt.xlabel(xName)
    plt.ylabel(yName)
    colorList = ['green', 'darkorange', 'blue', 'yellow']
    plt.xlim([min(np.min(state[:,0]), np.max(refstate[:,0])), max(np.max(state[:,0]), np.max(refstate[:,0]))])
    plt.ylim([min(np.min(state[:,1]), np.max(refstate[:,1])), max(np.max(state[:,1]), np.max(refstate[:,1]))])
    for step in range(state.shape[0]):
        plt.pause(1)
        plt.scatter(state[step][0], state[step][1], color='red', s=5)
        plt.scatter(refstate[step][0], refstate[step][1], color='blue', s=5)
    plt.pause(20)
    plt.ioff()
    plt.close()

if __name__ == '__main__':
    config = MPCConfig()
    MPCStep = config.MPCStep
    ADP_dir = './Results_dir/2022-03-29-11-07-59'
    # 1. 真实时域中MPC表现
    # MPC参考点更新按照真实参考轨迹
    # 测试MPC跟踪性能
    # simu_dir = "./Simulation_dir/simulationMPC" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # os.makedirs(simu_dir, exist_ok=True)
    # simulationMPC(MPCStep, simu_dir)

    # 2. 虚拟时域中MPC表现（开环控制）
    simu_dir = "./Simulation_dir/simulationOpen" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(simu_dir, exist_ok=True)
    simulationOpen(MPCStep, simu_dir)

    # 3. 单步ADP、MPC测试
    # simu_dir = ADP_dir + '/simulationOneStep' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # os.makedirs(simu_dir, exist_ok=True)
    # simulationOneStep(MPCStep, ADP_dir, simu_dir, stateNum=200)

    # 4. 真实时域ADP、MPC应用
    # simu_dir = ADP_dir + '/simulationReal' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # os.makedirs(simu_dir, exist_ok=True)
    # simulationReal(MPCStep, ADP_dir, simu_dir)

    # simu_dir = ADP_dir + '/simulationVirtual' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # os.makedirs(simu_dir, exist_ok=True)
    # simulationVirtual(MPCStep, ADP_dir, simu_dir)
