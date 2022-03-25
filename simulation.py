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

def simulationMPC(MPCStep, simu_dir):
    # 测试MPC跟踪性能
    env = TrackingEnv()
    env.seed(0)
    stateDim = env.stateDim
    actionDim = env.actionSpace.shape[0]
    solver = Solver()
    for mpcstep in MPCStep:
        print("----------------------Start Solving!----------------------")
        print("MPCStep: {}".format(mpcstep))
        # plt.ion()
        plt.figure(mpcstep)
        tempstate = env.initializeState(200)
        tempstate = tempstate[0].tolist() # 这里就取了一个，可以考虑从0开始取
        state = tempstate[-3:] + tempstate[:3]
        refState = tempstate[3:6]
        count = 0
        x = torch.linspace(0, 30*np.pi, 1000)
        y, _ = env.referenceCurve(x)
        plt.xlim(-5, 100)
        plt.ylim(-1.1, 1.1)
        plt.plot(x, y, color='gray')
        controlMPC = np.empty(0)
        stateMPC = np.empty(0)
        while(count < env.testStep):
            _, control = solver.MPCSolver(state, refState, mpcstep, isReal = True)
            stateMPC = np.append(stateMPC, np.array(state))
            stateMPC = np.append(stateMPC, np.array(refState))
            action = control[0].tolist()
            state = env.vehicleDynamic(
                state[0], state[1], state[2], state[3], state[4], state[5], action[0], action[1], MPCflag=1)
            refState = env.referenceFind(refState[0], MPCflag = 1)[0:3]
            plt.scatter(state[0], state[1], color='red', s=2)
            plt.scatter(refState[0], refState[1], color='blue', s=2)
            controlMPC = np.append(controlMPC, control[0])
            # plt.pause(0.05)
            count += 1
            if count % 10 == 0:
                print('count/totalStep: {}/{}'.format(count, env.testStep))
        plt.title('MPCStep:'+str(mpcstep))
        plt.savefig(simu_dir + '/OnlyMPCStep'+str(mpcstep)+'.png')
        # plt.ioff()
        plt.close()
        controlMPC = np.reshape(controlMPC, (-1, actionDim))
        stateMPC = np.reshape(stateMPC, (-1, stateDim))
        saveMPC = np.concatenate((stateMPC, controlMPC), axis = 1)
        with open(simu_dir + "/simulationMPC_" + str(mpcstep)+".csv", 'ab') as f:
            np.savetxt(f, saveMPC, delimiter=',', fmt='%.4f', comments='', header="x,y,phi,u,v,omega,xr,yr,phir,a,delta")

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
        x = torch.linspace(0, 30*np.pi, 1000)
        y, _ = env.referenceCurve(x)
        plt.xlim(-5, 100)
        plt.ylim(-1.1, 1.1)
        plt.plot(x, y, color='gray')
        count = 0
        _, control = solver.MPCSolver(state, refState, mpcstep, isReal=False)
        plt.scatter(state[0], state[1], color='red', s=5)
        plt.scatter(refState[0], refState[1], color='blue', s=5)
        while(count < mpcstep):
            action = control[count].tolist()
            state = env.vehicleDynamic(
                state[0], state[1], state[2], state[3], state[4], state[5], action[0], action[1], MPCflag=1)
            refState = env.refdynamicvirtual(refState, MPCflag=1)[:3]
            plt.scatter(state[0], state[1], color='red', s=5)
            plt.scatter(refState[0], refState[1], color='blue', s=5)
            count += 1
        plt.title('MPCStep:'+str(mpcstep))
        plt.savefig(simu_dir + '/simulationOpen'+str(mpcstep)+'.png')
        plt.close()
        
def simulationOneStep(MPCStep, ADP_dir, simu_dir, stateNum):
    # 单步ADP、MPC测试
    env = TrackingEnv()
    env.seed(0)
    relstateDim = env.relstateDim
    actionDim = env.actionSpace.shape[0]
    policy = Actor(relstateDim, actionDim)
    policy.loadParameters(ADP_dir)
    value = Critic(relstateDim, 1)
    value.loadParameters(ADP_dir)
    solver = Solver()
    initialState = env.initializeState(stateNum) # [u,v,omega,[xr,yr,phir],x,y,phi]
    timeStart = time.time()
    relState = env.relStateCal(initialState)
    controlADP = policy(relState).detach()
    timeADP = (time.time() - timeStart)
    controlADP = controlADP.numpy()
    print("ADP consumes {:.3f}s {} step".format(timeADP, stateNum))
    for mpcstep in MPCStep:
        timeMPC = 0
        controlMPC = np.empty(0)
        print("----------------------Start Solving MPC" +str(mpcstep)+"!----------------------")
        for i in range(stateNum):
            tempstate = initialState[i].tolist() # 这里就取了一个，可以考虑从0开始取
            state = tempstate[-3:] + tempstate[:3]
            refState = tempstate[3:6]
            timeStart = time.time()
            _, control = solver.MPCSolver(state, refState, mpcstep, isReal=False)
            timeMPC += time.time() - timeStart
            controlMPC = np.append(controlMPC, control[0])
        controlMPC = np.reshape(controlMPC, (-1, actionDim))
        print("MPC{} consumes {:.3f}s {} step".format(mpcstep, timeMPC, stateNum))
        # TODO: 这么做合适吗
        maxAction = np.array(env.actionHigh)
        minAction = np.array(env.actionLow)
        # maxAction = np.max(controlMPC, 0)
        # minACtion = np.min(controlMPC, 0)
        relativeError = np.abs(
            (controlADP - controlMPC)/(maxAction - minAction))
        relativeErrorMax = np.max(relativeError, 0)
        relativeErrorMean = np.mean(relativeError, 0)
        for i in range(actionDim):
            print('Relative error for action{}'.format(i+1))
            print('Mean: {:.2f}%, Max: {:.2f}'.format(relativeErrorMean[i]*100,relativeErrorMax[i]*100))
        saveMPC = np.concatenate((controlADP, controlMPC, relativeError), axis = 1)
        with open(simu_dir + "/simulationOneStepMPC_"+str(mpcstep)+".csv", 'ab') as f:
            np.savetxt(f, saveMPC, delimiter=',', fmt='%.4f', comments='', header="ADP_a,ADP_delta,MPC_a,MPC_delta,rel_a,rel_delta")
        plt.figure()
        data = relativeError[:, 0]
        plt.hist(data, bins=30, weights = np.zeros_like(data) + 1 / len(data))
        plt.xlabel('Relative Error')
        plt.ylabel('Frequency')
        plt.title('Relative Error of control a')
        plt.savefig(simu_dir + '/simulationOneStep'+str(mpcstep)+'_a.png')
        plt.close()
        plt.figure()
        data = relativeError[:, 1]
        plt.hist(data, bins=30, weights = np.zeros_like(data) + 1 / len(data))
        plt.xlabel('Relative Error')
        plt.ylabel('Frequency')
        plt.title('Relative Error of control delta')
        plt.savefig(simu_dir + '/simulationOneStep'+str(mpcstep)+'_delta.png')
        plt.close()


    def simulationReal(MPCStep, ADP_dir, simu_dir):
        # 真实时域ADP、MPC应用
        env = TrackingEnv()
        env.seed(0)
        stateDim = env.stateDim - 2
        actionDim = env.actionSpace.shape[0]
        policy = Actor(stateDim, actionDim)
        policy.loadParameters(ADP_dir)
        value = Critic(stateDim, 1)
        value.loadParameters(ADP_dir)
        solver = Solver()
        env.policyTest(policy, -1, simu_dir)
        for mpcstep in MPCStep:
            print("----------------------Start Solving!----------------------")
            print("MPCStep: {}".format(mpcstep))
            # plt.ion()
            plt.figure(mpcstep)
            state = env.initState
            count = 0
            x = torch.linspace(0, 30*np.pi, 1000)
            y = env.referenceCurve(x)
            plt.xlim(-5, 100)
            plt.ylim(-1.1, 1.1)
            plt.plot(x, y, color='gray')
            controlMPC = np.empty(0)
            stateMPC = np.empty(0)
            while(count < env.testStep):
                refState = env.referencePoint(state[0], MPCflag=1)
                _, control = solver.MPCSolver(state, refState, mpcstep)
                stateMPC = np.append(stateMPC, np.arry(state))
                stateMPC = np.append(stateMPC, np.array(refState))
                action = control[0].tolist()
                state = env.vehicleDynamic(
                    state[0], state[1], state[2], state[3], state[4], state[5], action[0], action[1], MPCflag=1)
                plt.scatter(state[0], state[1], color='red', s=2)
                plt.scatter(refState[0], refState[1], color='blue', s=2)
                controlMPC = np.append(controlMPC, control[0])
                # plt.pause(0.05)
                count += 1
                if count % 10 == 0:
                    print('count/totalStep: {}/{}'.format(count, env.testStep))
            plt.title('MPCStep:'+str(mpcstep))
            plt.savefig(simu_dir + '/FullMPCStep'+str(mpcstep)+'.png')
            # plt.ioff()
            plt.close()
            controlMPC = np.reshape(controlMPC, (-1, actionDim))
            np.savetxt(os.path.join(simu_dir, "controlFullMPC" +
                    str(mpcstep)+".csv"), controlMPC, delimiter=',')
            stateMPC = np.reshape(stateMPC, (-1, stateDim + 2))
            np.savetxt(os.path.join(simu_dir, "stateFullMPC" +
                    str(mpcstep)+".csv"), stateMPC, delimiter=',')


if __name__ == '__main__':
    config = MPCConfig()
    MPCStep = config.MPCStep
    ADP_dir = './Results_dir/2022-03-25-15-36-08'
    # 1. 真实时域中MPC表现
    # MPC参考点更新按照真实参考轨迹
    # 测试MPC跟踪性能
    # simu_dir = "./Simulation_dir/simulationMPC" + datetime.now().strftime("%Y-%m-%d-%H-%M")
    # os.makedirs(simu_dir, exist_ok=True)
    # simulationMPC(MPCStep, simu_dir)

    # 2. 虚拟时域中MPC表现（开环控制）
    # simu_dir = "./Simulation_dir/simulationOpen" + datetime.now().strftime("%Y-%m-%d-%H-%M")
    # os.makedirs(simu_dir, exist_ok=True)
    # simulationOpen(MPCStep, simu_dir)

    # 3. 单步ADP、MPC测试
    # simu_dir = ADP_dir + '/simulationOneStep' + datetime.now().strftime("%Y-%m-%d-%H-%M")
    # os.makedirs(simu_dir, exist_ok=True)
    # simulationOneStep(MPCStep, ADP_dir, simu_dir, stateNum=200)

    # 4. 真实时域ADP、MPC应用
    simu_dir = ADP_dir + '/simulationFull' + datetime.now().strftime("%Y-%m-%d-%H-%M")
    os.makedirs(simu_dir, exist_ok=True)
    simulationReal(MPCStep, ADP_dir, simu_dir)
