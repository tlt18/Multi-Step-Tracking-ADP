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


def simulationFull(MPCStep, ADP_dir, simu_dir):
    # MPC
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
    # MPC
    env = TrackingEnv()
    env.seed(0)
    stateDim = env.stateDim - 2
    actionDim = env.actionSpace.shape[0]
    policy = Actor(stateDim, actionDim)
    policy.loadParameters(ADP_dir)
    value = Critic(stateDim, 1)
    value.loadParameters(ADP_dir)
    solver = Solver()
    initialState = env.initializeState(stateNum)
    timeStart = time.time()
    controlADP = policy(initialState).detach()
    timeADP = (time.time() - timeStart)
    controlADP = controlADP.numpy()
    np.savetxt(os.path.join(simu_dir, "controlOneStepADP.csv"),
               controlADP, delimiter=',')
    print("ADP consumes {:.3f}s {} step".format(timeADP, stateNum))
    for mpcstep in MPCStep:
        timeMPC = 0
        controlMPC = np.empty((0, 2))
        # plt.ion()
        # plt.figure()
        # x = torch.linspace(0, 30*np.pi, 1000)
        # y = env.referenceCurve(x)
        # plt.xlim(-5, 100)
        # plt.ylim(-1.1, 1.1)
        # plt.plot(x, y, color='gray')
        print("----------------------Start Solving MPC" +
              str(mpcstep)+"!----------------------")
        for i in range(stateNum):
            state = initialState[i].tolist()
            timeStart = time.time()
            _, control = solver.MPCSolver(state[0:6], state[6:], mpcstep)
            timeMPC += time.time() - timeStart
            controlMPC = np.append(controlMPC, control[0])

            # plotX = [state[0], state[0] + 10 * math.cos(state[2])]
            # plotY = [state[1], state[1] + 10 * math.sin(state[2])]
            # plt.plot(plotX, plotY, color='red', linewidth=1)
            # plt.scatter(state[6], state[7], color='blue', s=5)
            # plt.title('x='+str(round(state[0], 1)) +
            #           ',y='+str(round(state[1], 1)) +
            #           ',delta='+str(round(control[0][1], 4))
            #           )
            # plt.pause(0.1)
            # TODO: 清除
            # plt.cla()
            # plt.plot(x, y, color='gray')
            # plt.xlim(-5, 100)
            # plt.ylim(-1.1, 1.1)
        # plt.close()
        # plt.ioff()
        timeMPC = timeMPC
        controlMPC = np.reshape(controlMPC, (-1, actionDim))
        print("MPC{} consumes {:.3f}s {} step".format(
            mpcstep, timeMPC, stateNum))
        np.savetxt(os.path.join(simu_dir, "controlOneStepMPC" +
                   str(mpcstep)+".csv"), controlMPC, delimiter=',')
        # maxAction = np.max(controlMPC, 0)
        # minACtion = np.min(controlMPC, 0)
        maxAction = np.array(env.actionHigh)
        minAction = np.array(env.actionLow)
        relativeError = np.abs(
            (controlADP - controlMPC)/(maxAction - minAction))
        relativeErrorMax = np.max(relativeError, 0)
        for i in range(actionDim):
            print('max relative error for action{}: {:.1f}%'.format(
                i+1, relativeErrorMax[i]*100))


# def simulationOneStep(MPCStep, ADP_dir, simu_dir, stateNum):
#     # MPC
#     env = TrackingEnv()
#     env.seed(0)
#     stateDim = env.stateDim - 2
#     actionDim = env.actionSpace.shape[0]
#     policy = Actor(stateDim, actionDim)
#     policy.loadParameters(ADP_dir)
#     value = Critic(stateDim, 1)
#     value.loadParameters(ADP_dir)
#     solver = Solver()
#     initialState = env.initializeState(stateNum)
#     timeStart = time.time()
#     controlADP = policy(initialState).detach()
#     timeADP = (time.time() - timeStart)
#     controlADP = controlADP.numpy()
#     np.savetxt(os.path.join(simu_dir, "controlOneStepADP.csv"),
#                controlADP, delimiter=',')
#     print("ADP consumes {:.3f}s {} step".format(timeADP, stateNum))
#     for mpcstep in MPCStep:
#         timeMPC = 0
#         controlMPC = np.empty((0, 2))
#         # plt.ion()
#         # plt.figure()
#         # x = torch.linspace(0, 30*np.pi, 1000)
#         # y = env.referenceCurve(x)
#         # plt.xlim(-5, 100)
#         # plt.ylim(-1.1, 1.1)
#         # plt.plot(x, y, color='gray')
#         print("----------------------Start Solving MPC" +
#               str(mpcstep)+"!----------------------")
#         for i in range(stateNum):
#             state = initialState[i].tolist()
#             timeStart = time.time()
#             _, control = solver.MPCSolver(state[0:6], state[6:], mpcstep)
#             timeMPC += time.time() - timeStart
#             controlMPC = np.append(controlMPC, control[0])

#             # plotX = [state[0], state[0] + 10 * math.cos(state[2])]
#             # plotY = [state[1], state[1] + 10 * math.sin(state[2])]
#             # plt.plot(plotX, plotY, color='red', linewidth=1)
#             # plt.scatter(state[6], state[7], color='blue', s=5)
#             # plt.title('x='+str(round(state[0], 1)) +
#             #           ',y='+str(round(state[1], 1)) +
#             #           ',delta='+str(round(control[0][1], 4))
#             #           )
#             # plt.pause(0.1)
#             # TODO: 清除
#             # plt.cla()
#             # plt.plot(x, y, color='gray')
#             # plt.xlim(-5, 100)
#             # plt.ylim(-1.1, 1.1)
#         # plt.close()
#         # plt.ioff()
#         timeMPC = timeMPC
#         controlMPC = np.reshape(controlMPC, (-1, actionDim))
#         print("MPC{} consumes {:.3f}s {} step".format(
#             mpcstep, timeMPC, stateNum))
#         np.savetxt(os.path.join(simu_dir, "controlOneStepMPC" +
#                    str(mpcstep)+".csv"), controlMPC, delimiter=',')
#         # maxAction = np.max(controlMPC, 0)
#         # minACtion = np.min(controlMPC, 0)
#         maxAction = np.array(env.actionHigh)
#         minAction = np.array(env.actionLow)
#         relativeError = np.abs(
#             (controlADP - controlMPC)/(maxAction - minAction))
#         relativeErrorMax = np.max(relativeError, 0)
#         for i in range(actionDim):
#             print('max relative error for action{}: {:.1f}%'.format(
#                 i+1, relativeErrorMax[i]*100))

if __name__ == '__main__':
    config = MPCConfig()
    MPCStep = config.MPCStep

    simu_dir = "./Simulation_dir/" + datetime.now().strftime("%Y-%m-%d-%H-%M")
    os.makedirs(simu_dir, exist_ok=True)

    # 真实时域中MPC表现
    # MPC参考点更新按照真实参考轨迹
    # 测试MPC跟踪性能
    # simulationMPC(MPCStep, simu_dir)

    # 虚拟时域中MPC表现（开环控制）
    # simulationOpen(MPCStep, simu_dir)

    ADP_dir = './Results_dir/2022-03-25-15-36-08'
    simu_dir = ADP_dir + '/' + datetime.now().strftime("%Y-%m-%d-%H-%M")
    os.makedirs(simu_dir, exist_ok=True)

    # 单步ADP、MPC测试
    simulationOneStep(MPCStep, ADP_dir, simu_dir, stateNum=200)

    # 真实时域ADP、MPC应用
    # simulationFull(MPCStep, ADP_dir, simu_dir)
