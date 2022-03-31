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
        while(count < env.testStepReal):
            _, control = solver.MPCSolver(state, refState, mpcstep, isReal = True)
            stateMPC = np.append(stateMPC, np.array(state))
            stateMPC = np.append(stateMPC, np.array(refState))
            action = control[0].tolist()
            state = env.vehicleDynamic(
                state[0], state[1], state[2], state[3], state[4], state[5], action[0], action[1], MPCflag=1)
            refState = env.refDynamicReal(refState[0], MPCflag = 1)[0:3]
            plt.scatter(state[0], state[1], color='red', s=2)
            plt.scatter(refState[0], refState[1], color='blue', s=2)
            controlMPC = np.append(controlMPC, control[0])
            # plt.pause(0.05)
            count += 1
            if count % 10 == 0:
                print('count/totalStep: {}/{}'.format(count, env.testStepReal))
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
        stateList = np.empty(0)
        refList = np.empty(0)
        stateList = np.append(stateList, state)
        refList = np.append(refList, refState)
        while(count < mpcstep):
            action = control[count].tolist()
            state = env.vehicleDynamic(
                state[0], state[1], state[2], state[3], state[4], state[5], action[0], action[1], MPCflag=1)
            refState = env.refDynamicVirtual(refState, MPCflag=1)[:3]
            stateList = np.append(stateList, state)
            refList = np.append(refList, refState)
            plt.scatter(state[0], state[1], color='red', s=5)
            plt.scatter(refState[0], refState[1], color='blue', s=5)
            count += 1
        plt.title('MPCStep:'+str(mpcstep))
        plt.savefig(simu_dir + '/simulationOpen'+str(mpcstep)+'.png')
        plt.close()
        stateList = np.reshape(stateList, (-1, 6))
        refList = np.reshape(refList, (-1, env.refNum * 3))
        if mpcstep==MPCStep[-1]:
            animationPlot(stateList[:,:2], refList[:,:2],'x', 'y')
        
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
        # minAction = np.min(controlMPC, 0)
        relativeError = np.abs(
            (controlADP - controlMPC)/(maxAction - minAction))
        relativeErrorMax = np.max(relativeError, 0)
        relativeErrorMean = np.mean(relativeError, 0)
        for i in range(actionDim):
            print('Relative error for action{}'.format(i+1))
            print('Mean: {:.2f}%, Max: {:.2f}%'.format(relativeErrorMean[i]*100,relativeErrorMax[i]*100))
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
    relstateDim = env.relstateDim
    actionDim = env.actionSpace.shape[0]
    policy = Actor(relstateDim, actionDim)
    policy.loadParameters(ADP_dir)
    value = Critic(relstateDim, 1)
    value.loadParameters(ADP_dir)
    solver = Solver()
    initialState = env.initializeState(1, isNoise = 0) # [u,v,omega,[xr,yr,phir],x,y,phi]

    # ADP
    stateAdp = initialState.clone()
    controlADPList = np.empty(0)
    stateADPList = np.empty(0)
    rewardADP = np.empty(0)
    count = 0
    while(count < env.testStepReal):
        stateADPList = np.append(stateADPList, stateAdp[0, -3:].numpy())
        stateADPList = np.append(stateADPList, stateAdp[0, :6].numpy())
        relState = env.relStateCal(stateAdp)
        controlAdp = policy(relState).detach()
        stateAdp, reward, done = env.stepReal(stateAdp, controlAdp)
        controlADPList = np.append(controlADPList, controlAdp[0].numpy())
        rewardADP = np.append(rewardADP, reward.numpy())
        count += 1
        stateADPList = np.reshape(stateADPList, (-1, env.stateDim))
        controlADPList = np.reshape(controlADPList, (-1, actionDim))
        saveADP = np.concatenate((stateADPList, controlADPList), axis = 1)
    with open(simu_dir + "/simulationRealADP.csv", 'ab') as f:
        np.savetxt(f, saveADP, delimiter=',', fmt='%.4f', comments='', header="x,y,phi,u,v,omega,xr,yr,phir,a,delta")
    controlMPCAll = []
    stateMPCAll = []
    rewardMPCAll = []
    for mpcstep in MPCStep:
        print("----------------------Start Solving!----------------------")
        print("MPCStep: {}".format(mpcstep))
        tempstate = initialState[0].tolist() # 这里就取了一个，可以考虑从0开始取
        stateMpc = tempstate[-3:] + tempstate[:3]
        refStateMpc = tempstate[3:6]
        count = 0
        controlMPCList = np.empty(0)
        stateMPCList = np.empty(0)
        rewardMPC = np.empty(0)
        while(count < env.testStepReal):
            # MPC
            _, control = solver.MPCSolver(stateMpc, refStateMpc, mpcstep, isReal = False)
            stateMPCList = np.append(stateMPCList, np.array(stateMpc))
            stateMPCList = np.append(stateMPCList, np.array(refStateMpc))
            action = control[0].tolist()
            reward = env.calReward(stateMpc[-3:] + refStateMpc + stateMpc[:3],action,MPCflag=1)
            stateMpc = env.vehicleDynamic(
                stateMpc[0], stateMpc[1], stateMpc[2], stateMpc[3], stateMpc[4], stateMpc[5], action[0], action[1], MPCflag=1)
            refStateMpc = env.refDynamicReal(refStateMpc[0], MPCflag=1)
            rewardMPC = np.append(rewardMPC, reward)
            controlMPCList = np.append(controlMPCList, control[0])
            count += 1
        stateMPCList = np.reshape(stateMPCList, (-1, env.stateDim))
        controlMPCList = np.reshape(controlMPCList, (-1, actionDim))
        saveMPC = np.concatenate((stateMPCList, controlMPCList), axis = 1)
        with open(simu_dir + "/simulationRealMPC_"+str(mpcstep)+".csv", 'ab') as f:
            np.savetxt(f, saveMPC, delimiter=',', fmt='%.4f', comments='', header="x,y,phi,u,v,omega,xr,yr,phir,a,delta")
        rewardMPCAll.append(rewardMPC)
        stateMPCAll.append(stateMPCList)
        controlMPCAll.append(controlMPCList)
        # print("Overall Cost for {} Steps, MPC: {:.3f}, ADP: {:.3f}".format(env.testStepReal, rewardMPC, rewardADP.item()))
        # y v.s. x
        plt.figure(mpcstep)
        x = torch.linspace(0, 30*np.pi, 1000)
        y, _ = env.referenceCurve(x)
        plt.xlim(-5, 100)
        plt.ylim(-1.1, 1.1)
        # MPC
        plt.scatter(stateMPCList[:,0], stateMPCList[:,1], s=2, c='blue', marker='o')
        # ADP
        plt.scatter(stateADPList[:,0], stateADPList[:,1], s=2, c='red', marker='*')
        # ref
        plt.scatter(stateMPCList[:,-3], stateMPCList[:,-2], c='gray', s = 2, marker='+')
        plt.legend(['MPC', 'ADP', 'reference'])
        plt.title('y-x ADP v.s. MPC-'+str(mpcstep))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(simu_dir + '/simulationRealADP_'+str(mpcstep)+'.png')
        plt.close()
        # 计算相对误差
        # maxAction = np.array(env.actionHigh)
        # minAction = np.array(env.actionLow)
        maxAction = np.max(controlMPCList, 0)
        minAction = np.min(controlMPCList, 0)
        relativeError = np.abs((controlADPList - controlMPCList)/(maxAction - minAction + 1e-3))
        relativeErrorMax = np.max(relativeError, 0)
        relativeErrorMean = np.mean(relativeError, 0)
        for i in range(actionDim):
            print('Relative error for action{}'.format(i+1))
            print('Mean: {:.2f}%, Max: {:.2f}%'.format(relativeErrorMean[i]*100,relativeErrorMax[i]*100))
        plt.figure()
        data = relativeError[:, 0]
        plt.hist(data, bins=30, weights = np.zeros_like(data) + 1 / len(data))
        plt.xlabel('Relative Error')
        plt.ylabel('Frequency')
        plt.title('Relative Error of control a')
        plt.savefig(simu_dir + '/relative error'+str(mpcstep)+'_a.png')
        plt.close()
        plt.figure()
        data = relativeError[:, 1]
        plt.hist(data, bins=30, weights = np.zeros_like(data) + 1 / len(data))
        # plt.xlim([0,0.1])
        plt.xlabel('Relative Error')
        plt.ylabel('Frequency')
        plt.title('Relative Error of control delta')
        plt.savefig(simu_dir + '/relative error'+str(mpcstep)+'_delta.png')
        plt.close()

    # x v.s. t
    MPCAll = [mpc[:,0] for mpc in stateMPCAll]
    ADP = stateADPList[:,0]
    xCoor = np.array(range(env.testStepReal)) * env.T
    xName = 'Time [s]'
    yName = 'x [m]'
    title = 'x-t ADP v.s. MPC'
    comparePlot(MPCAll, ADP, xCoor, MPCStep, xName, yName, simu_dir, title)
    # x error v.s. t
    MPCAll = [stateMPCAll[i][:,0]-stateMPCAll[i][:,-3] for i in range(len(stateMPCAll))]
    ADP = stateADPList[:,0] - stateADPList[:,-3]
    xCoor = np.array(range(env.testStepReal)) * env.T
    xName = 'Time [s]'
    yName = 'x error[m]'
    title = 'x error-t ADP v.s. MPC'
    comparePlot(MPCAll, ADP, xCoor, MPCStep, xName, yName, simu_dir, title)
    # y v.s. t
    MPCAll = [mpc[:,1] for mpc in stateMPCAll]
    ADP = stateADPList[:,1]
    xCoor = np.array(range(env.testStepReal)) * env.T
    xName = 'Time [s]'
    yName = 'y [m]'
    title = 'y-t ADP v.s. MPC'
    comparePlot(MPCAll, ADP, xCoor, MPCStep, xName, yName, simu_dir, title)
    # y error v.s. t
    MPCAll = [stateMPCAll[i][:,1]-stateMPCAll[i][:,-2] for i in range(len(stateMPCAll))]
    ADP = stateADPList[:,1] - stateADPList[:,-2]
    xCoor = np.array(range(env.testStepReal)) * env.T
    xName = 'Time [s]'
    yName = 'y error[m]'
    title = 'y error-t ADP v.s. MPC'
    comparePlot(MPCAll, ADP, xCoor, MPCStep, xName, yName, simu_dir, title)
    # phi v.s. t
    MPCAll = [mpc[:,2]* 180/np.pi for mpc in stateMPCAll] 
    ADP = stateADPList[:,2] * 180/np.pi
    xCoor = np.array(range(env.testStepReal)) * env.T
    xName = 'Time [s]'
    yName = 'phi [°]'
    title = 'phi-t ADP v.s. MPC'
    comparePlot(MPCAll, ADP, xCoor, MPCStep, xName, yName, simu_dir, title)
    # a v.s. t
    MPCAll = [mpc[:,0] for mpc in controlMPCAll]
    ADP = controlADPList[:,0]
    xCoor = np.array(range(env.testStepReal)) * env.T
    xName = 'Time [s]'
    yName = 'a [m/s^2]'
    title = 'a-t ADP v.s. MPC'
    comparePlot(MPCAll, ADP, xCoor, MPCStep, xName, yName, simu_dir, title)
    # delta v.s. t
    MPCAll = [mpc[:,1]* 180/np.pi for mpc in controlMPCAll]
    ADP = controlADPList[:,1] * 180/np.pi
    xCoor = np.array(range(env.testStepReal)) * env.T
    xName = 'Time [s]'
    yName = 'delta [°]'
    title = 'delta-t ADP v.s. MPC'
    comparePlot(MPCAll, ADP, xCoor, MPCStep, xName, yName, simu_dir, title)
    # utility v.s. t
    MPCAll = rewardMPCAll
    ADP = rewardADP
    xCoor = np.array(range(env.testStepReal)) * env.T
    xName = 'Time [s]'
    yName = 'Utility '
    title = 'utility-t ADP v.s. MPC'
    comparePlot(MPCAll, ADP, xCoor, MPCStep, xName, yName, simu_dir, title)
        # 动图
        # plt.figure(mpcstep)
        # plt.ion()
        # x = torch.linspace(0, 30*np.pi, 1000)
        # y, _ = env.referenceCurve(x)
        # plt.xlim(-5, 100)
        # plt.ylim(-1.1, 1.1)
        # plt.plot(x, y, color='gray', linewidth=1)
        # plt.title('Compare ADP v.s. MPC-'+str(mpcstep))
        # for i in range(env.testStepReal):
        #     # MPC
        #     plt.scatter(stateMPCList[i,0], stateMPCList[i,1], s=5, color='blue')
        #     # ADP
        #     plt.scatter(stateADPList[i,0], stateADPList[i,1], s=5, color='red')
        #     # ref
        #     plt.scatter(stateMPCList[i,-3], stateMPCList[i,-2], s=5, color='green')
        #     plt.pause(0.05)
        # plt.ioff()
        # plt.close()
    
def simulationVirtual(MPCStep, ADP_dir, simu_dir):
    # 真实时域ADP、MPC应用
    env = TrackingEnv()
    env.seed(0)
    relstateDim = env.relstateDim
    actionDim = env.actionSpace.shape[0]
    policy = Actor(relstateDim, actionDim)
    policy.loadParameters(ADP_dir)
    value = Critic(relstateDim, 1)
    value.loadParameters(ADP_dir)
    solver = Solver()
    initialState = env.initializeState(1, isNoise = 0) # [u,v,omega,[xr,yr,phir],x,y,phi]

    # ADP
    stateAdp = initialState.clone()
    controlADPList = np.empty(0)
    stateADPList = np.empty(0)
    rewardADP = np.empty(0)
    count = 0
    while(count < max(MPCStep)):
        stateADPList = np.append(stateADPList, stateAdp[0, -3:].numpy())
        stateADPList = np.append(stateADPList, stateAdp[0, :6].numpy())
        relState = env.relStateCal(stateAdp)
        controlAdp = policy(relState).detach()
        stateAdp, reward, done = env.stepVirtual(stateAdp, controlAdp)
        controlADPList = np.append(controlADPList, controlAdp[0].numpy())
        rewardADP = np.append(rewardADP, reward.numpy())
        count += 1
        stateADPList = np.reshape(stateADPList, (-1, env.stateDim))
        controlADPList = np.reshape(controlADPList, (-1, actionDim))
        saveADP = np.concatenate((stateADPList, controlADPList), axis = 1)
    with open(simu_dir + "/simulationRealADP.csv", 'ab') as f:
        np.savetxt(f, saveADP, delimiter=',', fmt='%.4f', comments='', header="x,y,phi,u,v,omega,xr,yr,phir,a,delta")
    controlMPCAll = []
    stateMPCAll = []
    rewardMPCAll = []
    for mpcstep in MPCStep:
        print("----------------------Start Solving!----------------------")
        print("MPCStep: {}".format(mpcstep))
        tempstate = initialState[0].tolist() # 这里就取了一个，可以考虑从0开始取
        stateMpc = tempstate[-3:] + tempstate[:3] # (x,y,phi,u,v,omega)
        refStateMpc = tempstate[3:6]
        count = 0
        controlMPCList = np.empty(0)
        stateMPCList = np.empty(0)
        rewardMPC = np.empty(0)
        _, control = solver.MPCSolver(stateMpc, refStateMpc, mpcstep, isReal=False)
        while(count < mpcstep):
            stateMPCList = np.append(stateMPCList, np.array(stateMpc))
            stateMPCList = np.append(stateMPCList, np.array(refStateMpc))
            action = control[count].tolist()
            reward = env.calReward(stateMpc[-3:] + refStateMpc + stateMpc[:3],action,MPCflag=1)
            stateMpc = env.vehicleDynamic(
                stateMpc[0], stateMpc[1], stateMpc[2], stateMpc[3], stateMpc[4], stateMpc[5], action[0], action[1], MPCflag=1)
            refStateMpc = env.refDynamicVirtual(refStateMpc, MPCflag=1)
            rewardMPC = np.append(rewardMPC, reward)
            controlMPCList = np.append(controlMPCList, control[count])
            count += 1
        stateMPCList = np.reshape(stateMPCList, (-1, env.stateDim))
        controlMPCList = np.reshape(controlMPCList, (-1, actionDim))
        saveMPC = np.concatenate((stateMPCList, controlMPCList), axis = 1)
        with open(simu_dir + "/simulationVirtualADP_"+str(mpcstep)+".csv", 'ab') as f:
            np.savetxt(f, saveMPC, delimiter=',', fmt='%.4f', comments='', header="x,y,phi,u,v,omega,xr,yr,phir,a,delta")
        rewardMPCAll.append(rewardMPC)
        stateMPCAll.append(stateMPCList)
        controlMPCAll.append(controlMPCList)
        # print("Overall Cost for {} Steps, MPC: {:.3f}, ADP: {:.3f}".format(env.testStepVirtual, rewardMPC, rewardADP.item()))
        # y v.s. x
        plt.figure(mpcstep)
        plt.scatter(stateMPCList[:mpcstep,0], stateMPCList[:mpcstep,1], s=20, c='blue', marker='o')
        # ADP
        plt.scatter(stateADPList[:mpcstep,0], stateADPList[:mpcstep,1], s=20, c='red', marker='*')
        # ref
        plt.scatter(stateMPCList[:,-3], stateMPCList[:,-2], c='gray', s = 20, marker='+')
        plt.legend(labels = ['MPC', 'ADP', 'reference'])
        plt.title('y-x ADP v.s. MPC-'+str(mpcstep))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(simu_dir + '/simulationVirtualMPC_'+str(mpcstep)+'.png')
        plt.close()
        # 计算相对误差
        # maxAction = np.array(env.actionHigh)
        # minAction = np.array(env.actionLow)
        maxAction = np.max(controlMPCList, 0)
        minAction = np.min(controlMPCList, 0)
        relativeError = np.abs((controlADPList[:mpcstep] - controlMPCList)/(maxAction - minAction + 1e-3))
        relativeErrorMax = np.max(relativeError, 0)
        relativeErrorMean = np.mean(relativeError, 0)
        for i in range(actionDim):
            print('Relative error for action{}'.format(i+1))
            print('Mean: {:.2f}%, Max: {:.2f}%'.format(relativeErrorMean[i]*100,relativeErrorMax[i]*100))
        plt.figure()
        data = relativeError[:, 0]
        plt.hist(data, bins=30, weights = np.zeros_like(data) + 1 / len(data))
        plt.xlabel('Relative Error')
        plt.ylabel('Frequency')
        plt.title('Relative Error of control a')
        plt.savefig(simu_dir + '/relative error'+str(mpcstep)+'_a.png')
        plt.close()
        plt.figure()
        data = relativeError[:, 1]
        plt.hist(data, bins=30, weights = np.zeros_like(data) + 1 / len(data))
        # plt.xlim([0,0.1])
        plt.xlabel('Relative Error')
        plt.ylabel('Frequency')
        plt.title('Relative Error of control delta')
        plt.savefig(simu_dir + '/relative error'+str(mpcstep)+'_delta.png')
        plt.close()

    # x v.s. t
    MPCAll = [mpc[:,0] for mpc in stateMPCAll]
    ADP = stateADPList[:,0]
    xCoor = np.array(range(max(MPCStep))) * env.T
    xName = 'Time [s]'
    yName = 'x [m]'
    title = 'x-t ADP v.s. MPC'
    comparePlot(MPCAll, ADP, xCoor, MPCStep, xName, yName, simu_dir, title)
    # x error v.s. t
    MPCAll = [stateMPCAll[i][:,0]-stateMPCAll[i][:,-3] for i in range(len(stateMPCAll))]
    ADP = stateADPList[:,0] - stateADPList[:,-3]
    xCoor = np.array(range(max(MPCStep))) * env.T
    xName = 'Time [s]'
    yName = 'x error[m]'
    title = 'x error-t ADP v.s. MPC'
    comparePlot(MPCAll, ADP, xCoor, MPCStep, xName, yName, simu_dir, title)
    # y v.s. t
    MPCAll = [mpc[:,1] for mpc in stateMPCAll]
    ADP = stateADPList[:,1]
    xCoor = np.array(range(max(MPCStep))) * env.T
    xName = 'Time [s]'
    yName = 'y [m]'
    title = 'y-t ADP v.s. MPC'
    comparePlot(MPCAll, ADP, xCoor, MPCStep, xName, yName, simu_dir, title)
    # y error v.s. t
    MPCAll = [stateMPCAll[i][:,1]-stateMPCAll[i][:,-2] for i in range(len(stateMPCAll))]
    ADP = stateADPList[:,1] - stateADPList[:,-2]
    xCoor = np.array(range(max(MPCStep))) * env.T
    xName = 'Time [s]'
    yName = 'y error[m]'
    title = 'y error-t ADP v.s. MPC'
    comparePlot(MPCAll, ADP, xCoor, MPCStep, xName, yName, simu_dir, title)
    # phi v.s. t
    MPCAll = [mpc[:,2]* 180/np.pi for mpc in stateMPCAll] 
    ADP = stateADPList[:,2] * 180/np.pi
    xCoor = np.array(range(max(MPCStep))) * env.T
    xName = 'Time [s]'
    yName = 'phi [°]'
    title = 'phi-t ADP v.s. MPC'
    comparePlot(MPCAll, ADP, xCoor, MPCStep, xName, yName, simu_dir, title)
    # a v.s. t
    MPCAll = [mpc[:,0] for mpc in controlMPCAll]
    ADP = controlADPList[:,0]
    xCoor = np.array(range(max(MPCStep))) * env.T
    xName = 'Time [s]'
    yName = 'a [m/s^2]'
    title = 'a-t ADP v.s. MPC'
    comparePlot(MPCAll, ADP, xCoor, MPCStep, xName, yName, simu_dir, title)
    # delta v.s. t
    MPCAll = [mpc[:,1]* 180/np.pi for mpc in controlMPCAll]
    ADP = controlADPList[:,1] * 180/np.pi
    xCoor = np.array(range(max(MPCStep))) * env.T
    xName = 'Time [s]'
    yName = 'delta [°]'
    title = 'delta-t ADP v.s. MPC'
    comparePlot(MPCAll, ADP, xCoor, MPCStep, xName, yName, simu_dir, title)
    # utility v.s. t
    MPCAll = rewardMPCAll
    ADP = rewardADP
    xCoor = np.array(range(max(MPCStep))) * env.T
    xName = 'Time [s]'
    yName = 'Utility '
    title = 'utility-t ADP v.s. MPC'
    comparePlot(MPCAll, ADP, xCoor, MPCStep, xName, yName, simu_dir, title)

def comparePlot(MPCAll, ADP, xCoor, MPCStep, xName, yName, simu_dir, title):
    plt.figure()
    colorList = ['green', 'darkorange', 'blue', 'yellow']
    for i in range(len(MPCAll)):
        MPC = MPCAll[i]
        plt.plot(xCoor[:MPC.shape[0]], MPC, linewidth=2, color = colorList[i], linestyle = '-.')
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
    # 检查一下reward是否一样
    ADP_dir = './Results_dir/2022-03-30-19-40-49'
    # 1. 真实时域中MPC表现
    # MPC参考点更新按照真实参考轨迹
    # 测试MPC跟踪性能
    # simu_dir = "./Simulation_dir/simulationMPC" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # os.makedirs(simu_dir, exist_ok=True)
    # simulationMPC(MPCStep, simu_dir)

    # 2. 虚拟时域中MPC表现（开环控制）
    # simu_dir = "./Simulation_dir/simulationOpen" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # os.makedirs(simu_dir, exist_ok=True)
    # simulationOpen(MPCStep, simu_dir)

    # 3. 单步ADP、MPC测试
    # simu_dir = ADP_dir + '/simulationOneStep' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # os.makedirs(simu_dir, exist_ok=True)
    # simulationOneStep(MPCStep, ADP_dir, simu_dir, stateNum=200)

    # 4. 真实时域ADP、MPC应用
    simu_dir = ADP_dir + '/simulationReal' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    simu_dir = ADP_dir + '/simulationReal'
    os.makedirs(simu_dir, exist_ok=True)
    simulationReal(MPCStep, ADP_dir, simu_dir)

    # 5. 虚拟时域ADP、MPC应用
    # simu_dir = ADP_dir + '/simulationVirtual' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    simu_dir = ADP_dir + '/simulationVirtual'
    os.makedirs(simu_dir, exist_ok=True)
    simulationVirtual(MPCStep, ADP_dir, simu_dir)
