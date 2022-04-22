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
    plotDelete = 0
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
    initialState = env.resetSpecificCurve(1, curveType = 'sine', noise = 0) # [u,v,omega,[xr,yr,phir],x,y,phi]

    # ADP
    stateAdp = initialState.clone()
    controlADPList = np.empty(0)
    stateADPList = np.empty(0)
    rewardADP = np.empty(0)
    count = 0
    while(count < env.testStepReal):
        stateADPList = np.append(stateADPList, stateAdp[0, -3:].numpy()) # x, y, phi
        stateADPList = np.append(stateADPList, stateAdp[0, :-3].numpy()) # u, v, omega, [xr, yr, phir]
        relState = env.relStateCal(stateAdp)
        controlAdp = policy(relState).detach()
        stateAdp, reward, done = env.stepReal(stateAdp, controlAdp, curveType = 'sine')
        controlADPList = np.append(controlADPList, controlAdp[0].numpy())
        rewardADP = np.append(rewardADP, reward.numpy())
        count += 1

    stateADPList = np.reshape(stateADPList, (-1, env.stateDim))
    controlADPList = np.reshape(controlADPList, (-1, actionDim))

    stateADPList = np.delete(stateADPList, range(plotDelete), 0)
    controlADPList = np.delete(controlADPList, range(plotDelete), 0)
    rewardADP = np.delete(rewardADP, range(plotDelete), 0)

    saveADP = np.concatenate((stateADPList, controlADPList), axis = 1)
    with open(simu_dir + "/simulationRealADP.csv", 'wb') as f:
        np.savetxt(f, saveADP, delimiter=',', fmt='%.4f', comments='', header="x,y,phi,u,v,omega," + "xr,yr,phir,"*env.refNum + "a,delta")

    controlMPCAll = []
    stateMPCAll = []
    rewardMPCAll = []
    for mpcstep in MPCStep:
        print("----------------------Start Solving!----------------------")
        print("MPCStep: {}".format(mpcstep))
        tempstate = initialState[0].tolist()
        stateMpc = tempstate[-3:] + tempstate[:3] # x, y, phi, u, v, omega
        refStateMpc = tempstate[3:-3]
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
            refStateMpc = env.refDynamicReal(refStateMpc, MPCflag=1, curveType = 'sine')
            rewardMPC = np.append(rewardMPC, reward)
            controlMPCList = np.append(controlMPCList, control[0])
            count += 1
        stateMPCList = np.reshape(stateMPCList, (-1, env.stateDim))
        controlMPCList = np.reshape(controlMPCList, (-1, actionDim))
        stateMPCList = np.delete(stateMPCList, range(plotDelete), 0)
        controlMPCList = np.delete(controlMPCList, range(plotDelete), 0)
        rewardMPC = np.delete(rewardMPC, range(plotDelete), 0)

        saveMPC = np.concatenate((stateMPCList, controlMPCList), axis = 1)
        with open(simu_dir + "/simulationRealMPC_"+str(mpcstep)+".csv", 'wb') as f:
            np.savetxt(f, saveMPC, delimiter=',', fmt='%.4f', comments='', header="x,y,phi,u,v,omega," + "xr,yr,phir,"*env.refNum + "a,delta")
        rewardMPCAll.append(rewardMPC)
        stateMPCAll.append(stateMPCList)
        controlMPCAll.append(controlMPCList)
        # print("Overall Cost for {} Steps, MPC: {:.3f}, ADP: {:.3f}".format(env.testStepReal, rewardMPC, rewardADP.item()))

    # stateADPList: [x,y,phi,u,v,omega,[xr,yr,phir]]
    # controlMPCAll: [a, delta]
    # Cal relative error
    errorSaveList = []
    # Acceleration
    ADPData = controlADPList[:,0]
    MPCData = controlMPCAll[-1][:, 0]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Acceleration', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    # Steering Angle
    ADPData = controlADPList[:,1]
    MPCData = controlMPCAll[-1][:, 1]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Steering Angle', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    # Distance Error
    ADPData = np.sqrt(np.power(stateADPList[:, 0] - stateADPList[:, 6], 2) + np.power(stateADPList[:, 1] - stateADPList[:, 7], 2))
    MPCData = np.sqrt(np.power(stateMPCAll[-1][:, 0] - stateMPCAll[-1][:, 6], 2) + np.power(stateMPCAll[-1][:, 1] - stateMPCAll[-1][:, 7], 2))
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Distance Error', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    # Heading Angle
    ADPData = stateADPList[:,2] - stateADPList[:,8]
    MPCData = stateMPCAll[-1][:,2] - stateMPCAll[-1][:,8]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Heading Angle Error', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)
    
    # Utility  Function
    ADPData = rewardADP
    MPCData = rewardMPCAll[-1]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Utility  Function', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    if os.path.exists(simu_dir + "/RelError.csv")==False:
        with open(simu_dir + "/RelError.csv", 'ab') as f:
            np.savetxt(f, np.array([errorSaveList]), delimiter=',', fmt='%.4f', comments='', \
                header='Acceleration mean,max,Steering Angle mean,max,Distance Error mean,max,Heading Angle mean,max,Utility Function mean, max')
    else:
        with open(simu_dir + "/RelError.csv", 'ab') as f:
            np.savetxt(f, np.array([errorSaveList]), delimiter=',', fmt='%.4f', comments='')

    # Plot
    # stateAll: [x,y,phi,u,v,omega,[xr,yr,phir]]
    # controlAll: [a, delta]
    # y v.s. x
    xADP = stateADPList[:,0]
    xMPC = [mpc[:,0] for mpc in stateMPCAll]
    yADP = stateADPList[:,1]
    yMPC = [mpc[:,1] for mpc in stateMPCAll]
    xName = 'x position [m]'
    yName = 'y position [m]'
    title = 'y-x'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title)

    # distance error v.s. x
    xADP = stateADPList[:,0]
    xMPC = [mpc[:,0] for mpc in stateMPCAll]
    yADP = np.sqrt(np.power(stateADPList[:, 0] - stateADPList[:, 6], 2) + np.power(stateADPList[:, 1] - stateADPList[:, 7], 2))
    yMPC = [np.sqrt(np.power(mpc[:, 0] - mpc[:, 6], 2) + np.power(mpc[:, 1] - mpc[:, 7], 2)) for mpc in stateMPCAll]
    xName = 'Travel dist [m]'
    yName = 'Distance error [m]'
    title = 'distance-error-x'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isError = True)

    # phi v.s. x
    xADP = stateADPList[:,0]
    xMPC = [mpc[:,0] for mpc in stateMPCAll]
    yADP = stateADPList[:,2] * 180/np.pi
    yMPC = [mpc[:,2] * 180/np.pi for mpc in stateMPCAll]
    xName = 'Travel dist [m]'
    yName = 'Heading angle [°]'
    title = 'phi-x'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title)

    # phi error v.s. x
    xADP = stateADPList[:,0]
    xMPC = [mpc[:,0] for mpc in stateMPCAll]
    yADP = stateADPList[:,2] * 180/np.pi - stateADPList[:,8] * 180/np.pi
    yMPC = [mpc[:,2] * 180/np.pi - mpc[:,8] * 180/np.pi for mpc in stateMPCAll]
    xName = 'Travel dist [m]'
    yName = 'Heading angle error [°]'
    title = 'phi-error-x'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isError = True)

    # utility v.s. x
    xADP = stateADPList[:,0]
    xMPC = [mpc[:,0] for mpc in stateMPCAll]
    yADP = rewardADP
    yMPC = [mpc for mpc in rewardMPCAll]
    xName = 'Travel dist [m]'
    yName = 'utility'
    title = 'utility-x'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title)

    # a v.s. x
    xADP = stateADPList[:,0]
    xMPC = [mpc[:,0] for mpc in stateMPCAll]
    yADP = controlADPList[:,0] * 180/np.pi
    yMPC = [mpc[:,0] * 180/np.pi for mpc in controlMPCAll]
    xName = 'Travel dist [m]'
    yName = 'Acceleration [m/s^2]'
    title = 'a-x'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title)

    # delta v.s. x
    xADP = stateADPList[:,0]
    xMPC = [mpc[:,0] for mpc in stateMPCAll]
    yADP = controlADPList[:,1] * 180/np.pi
    yMPC = [mpc[:,1] * 180/np.pi for mpc in controlMPCAll]
    xName = 'Travel dist [m]'
    yName = 'Steering angle [°]'
    title = 'delta-x'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title)
    
 
def simulationVirtual(MPCStep, ADP_dir, simu_dir, noise = 0, seed = 0):
    # 真实时域ADP、MPC应用
    plotDelete = 0
    env = TrackingEnv()
    env.seed(seed)
    relstateDim = env.relstateDim
    actionDim = env.actionSpace.shape[0]
    policy = Actor(relstateDim, actionDim)
    policy.loadParameters(ADP_dir)
    value = Critic(relstateDim, 1)
    value.loadParameters(ADP_dir)
    solver = Solver()
    initialState = env.resetRandom(1, noise = noise, MPCtest = True) # [u,v,omega,[xr,yr,phir],x,y,phi]

    # ADP
    stateAdp = initialState.clone()
    controlADPList = np.empty(0)
    stateADPList = np.empty(0)
    rewardADP = np.empty(0)
    count = 0
    while(count < max(MPCStep)):
        stateADPList = np.append(stateADPList, stateAdp[0, -3:].numpy()) # x, y, phi
        stateADPList = np.append(stateADPList, stateAdp[0, :-3].numpy()) # u, v, omega, [xr, yr, phir]
        relState = env.relStateCal(stateAdp)
        controlAdp = policy(relState).detach()
        stateAdp, reward, done = env.stepVirtual(stateAdp, controlAdp)
        controlADPList = np.append(controlADPList, controlAdp[0].numpy())
        rewardADP = np.append(rewardADP, reward.numpy())
        count += 1
    stateADPList = np.reshape(stateADPList, (-1, env.stateDim))
    controlADPList = np.reshape(controlADPList, (-1, actionDim))
    stateADPList = np.delete(stateADPList, range(plotDelete), 0)
    controlADPList = np.delete(controlADPList, range(plotDelete), 0)
    rewardADP = np.delete(rewardADP, range(plotDelete), 0)
    saveADP = np.concatenate((stateADPList, controlADPList), axis = 1)
    with open(simu_dir + "/simulationRealADP.csv", 'wb') as f:
        np.savetxt(f, saveADP, delimiter=',', fmt='%.4f', comments='', header="x,y,phi,u,v,omega," + "xr,yr,phir,"*env.refNum + "a,delta")

    # MPC
    controlMPCAll = []
    stateMPCAll = []
    rewardMPCAll = []
    print("----------------------Start Solving!----------------------")
    for mpcstep in MPCStep:
        # print("MPCStep: {}".format(mpcstep))
        tempstate = initialState[0].tolist()
        stateMpc = tempstate[-3:] + tempstate[:3] # [x,y,phi,u,v,omega]
        refStateMpc = tempstate[3:-3]
        count = 0
        controlMPCList = np.empty(0) # [a, delta]
        stateMPCList = np.empty(0) # [x,y,phi,u,v,omega,[xr,yr,phir]]
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
        stateMPCList = np.delete(stateMPCList, range(plotDelete), 0)
        controlMPCList = np.delete(controlMPCList, range(plotDelete), 0)
        rewardMPC = np.delete(rewardMPC, range(plotDelete), 0)
        saveMPC = np.concatenate((stateMPCList, controlMPCList), axis = 1)
        with open(simu_dir + "/simulationVirtualADP_"+str(mpcstep)+".csv", 'wb') as f:
            np.savetxt(f, saveMPC, delimiter=',', fmt='%.4f', comments='', header="x,y,phi,u,v,omega," + "xr,yr,phir,"*env.refNum + "a,delta")
        rewardMPCAll.append(rewardMPC)
        stateMPCAll.append(stateMPCList)
        controlMPCAll.append(controlMPCList)

    # controlADPList: [x,y,phi,u,v,omega,[xr,yr,phir]]
    # controlADPList: [a, delta]
    # Cal relative error
    errorSaveList = []
    # Acceleration
    ADPData = controlADPList[:,0]
    MPCData = controlMPCAll[-1][:, 0]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Acceleration', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    # Steering Angle
    ADPData = controlADPList[:,1]
    MPCData = controlMPCAll[-1][:, 1]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Steering Angle', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    # Distance Error
    ADPData = np.sqrt(np.power(stateADPList[:, 0] - stateADPList[:, 6], 2) + np.power(stateADPList[:, 1] - stateADPList[:, 7], 2))
    MPCData = np.sqrt(np.power(stateMPCAll[-1][:, 0] - stateMPCAll[-1][:, 6], 2) + np.power(stateMPCAll[-1][:, 1] - stateMPCAll[-1][:, 7], 2))
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Distance Error', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    # Heading Angle
    ADPData = stateADPList[:,2] - stateADPList[:,8]
    MPCData = stateMPCAll[-1][:,2] - stateMPCAll[-1][:,8]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Heading Angle Error', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)
    
    # Utility  Function
    ADPData = rewardADP
    MPCData = rewardMPCAll[-1]
    relativeErrorMean, relativeErrorMax = calRelError(ADPData, MPCData, title = 'Utility  Function', simu_dir = simu_dir)
    errorSaveList.append(relativeErrorMean)
    errorSaveList.append(relativeErrorMax)

    if os.path.exists(simu_dir + "/RelError.csv")==False:
        with open(simu_dir + "/RelError.csv", 'ab') as f:
            np.savetxt(f, np.array([errorSaveList]), delimiter=',', fmt='%.4f', comments='', \
                header='Acceleration mean,max,Steering Angle mean,max,Distance Error mean,max,Heading Angle mean,max,Utility Function mean, max')
    else:
        with open(simu_dir + "/RelError.csv", 'ab') as f:
            np.savetxt(f, np.array([errorSaveList]), delimiter=',', fmt='%.4f', comments='')
    
    # Plot [x,y,phi,u,v,omega,[xr,yr,phir]]
    # a v.s. t
    xADP = np.arange(0, len(controlADPList[:,0]) * env.T, env.T)
    xMPC = [np.arange(0, len(mpc[:,0]) * env.T, env.T) for mpc in controlMPCAll]
    yADP = controlADPList[:,0]
    yMPC = [mpc[:,0] for mpc in controlMPCAll]
    xName = 'Predictive horizon [s]'
    yName = 'Acceleration [m/s^2]'
    title = 'a-t'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isError = True)

    # delta v.s. t
    xADP = np.arange(0, len(controlADPList[:,0]) * env.T, env.T)
    xMPC = [np.arange(0, len(mpc[:,1]) * env.T, env.T) for mpc in controlMPCAll]
    yADP = controlADPList[:,1]
    yMPC = [mpc[:,1] for mpc in controlMPCAll]
    xName = 'Predictive horizon [s]'
    yName = 'Steering Angle [m/s^2]'
    title = 'delta-t'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isError = True)

    # distance error v.s. t
    yADP = np.sqrt(np.power(stateADPList[:, 0] - stateADPList[:, 6], 2) + np.power(stateADPList[:, 1] - stateADPList[:, 7], 2))
    yMPC = [np.sqrt(np.power(mpc[:, 0] - mpc[:, 6], 2) + np.power(mpc[:, 1] - mpc[:, 7], 2)) for mpc in stateMPCAll]
    xADP = np.arange(0, len(yADP) * env.T, env.T)
    xMPC = [np.arange(0, len(mpc) * env.T, env.T) for mpc in yMPC]
    xName = 'Predictive horizon [s]'
    yName = 'Distance error [m]'
    title = 'distance-error-t'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isError = True)

    # phi error v.s. t
    yADP = (stateADPList[:,2] - stateADPList[:,8]) * 180 / np.pi
    yMPC = [(mpc[:,2] - mpc[:,8]) * 180 / np.pi for mpc in stateMPCAll]
    xADP = np.arange(0, len(yADP) * env.T, env.T)
    xMPC = [np.arange(0, len(mpc) * env.T, env.T) for mpc in yMPC]
    xName = 'Predictive horizon [s]'
    yName = 'Heading angle error [m]'
    title = 'phi-error-t'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isError = True)

    # phi error v.s. distance error
    xADP = np.sqrt(np.power(stateADPList[:, 0] - stateADPList[:, 6], 2) + np.power(stateADPList[:, 1] - stateADPList[:, 7], 2))
    xMPC = [np.sqrt(np.power(mpc[:, 0] - mpc[:, 6], 2) + np.power(mpc[:, 1] - mpc[:, 7], 2)) for mpc in stateMPCAll]
    yADP = (stateADPList[:,2] - stateADPList[:,8]) * 180 / np.pi
    yMPC = [(mpc[:,2] - mpc[:,8]) * 180 / np.pi for mpc in stateMPCAll]
    xName = 'Distance error [m]'
    yName = 'Heading angle error [°]'
    title = 'phi-error-Distance-error'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isError = True)

    # y v.s. x
    xADP = stateADPList[:, 0]
    xMPC = [mpc[:, 0] for mpc in stateMPCAll]
    yADP = stateADPList[:, 1]
    yMPC = [mpc[:, 1] for mpc in stateMPCAll]
    xName = 'x posision [m]'
    yName = 'y posision [m]'
    title = 'y-x'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True)

    # utility v.s. t
    yADP = rewardADP
    yMPC = [mpc for mpc in rewardMPCAll]
    xADP = np.arange(0, len(yADP) * env.T, env.T)
    xMPC = [np.arange(0, len(mpc) * env.T, env.T) for mpc in yMPC]
    xName = 'Predictive horizon [s]'
    yName = 'utility'
    title = 'utility-t'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isError = True)

def comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = False, isError = False):
    plt.figure()
    colorList = ['darkorange', 'green', 'blue', 'yellow', 'red']
    if isMark == True:
        markerList = ['|', 'D', 'o', 'x', '*']
    else:
        markerList = ['None', 'None', 'None', 'None']
    for i in range(len(xMPC)):
        plt.plot(xMPC[i], yMPC[i], linewidth=2, color = colorList[i], linestyle = '--', marker=markerList[i], markersize=4)
    plt.plot(xADP, yADP, linewidth = 2, color=colorList[-1],linestyle = '--', marker=markerList[-1], markersize=4)
    if isError == True:
        plt.plot([np.min(xADP), np.max(xADP)], [0,0], linewidth = 1, color = 'grey', linestyle = '--')
        plt.legend(labels=['MPC'+str(mpcStep) for mpcStep in MPCStep] + ['ADP', 'Ref'])
    else:
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

def calRelError(ADP, MPC, title, simu_dir, isPlot = False):
    maxMPC = np.max(MPC, 0)
    minMPC = np.min(MPC, 0)
    relativeError = np.abs((ADP - MPC)/(maxMPC - minMPC + 1e-3))
    relativeErrorMax = np.max(relativeError, 0)
    relativeErrorMean = np.mean(relativeError, 0)
    print(title +' Error | Mean: {:.4f}%, Max: {:.4f}%'.format(relativeErrorMean*100,relativeErrorMax*100))
    if isPlot == True:
        plt.figure()
        data = relativeError
        plt.hist(data, bins=30, weights = np.zeros_like(data) + 1 / len(data))
        plt.xlabel('Relative Error of '+title)
        plt.ylabel('Frequency')
        plt.title('Relative Error of '+title)
        plt.savefig(simu_dir + '/relative-error-'+title+'.png')
        plt.close()
    return relativeErrorMean, relativeErrorMax

def main(ADP_dir):
    config = MPCConfig()
    MPCStep = config.MPCStep
    # 检查一下reward是否一样
    
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
    simu_dir = ADP_dir + '/simulationReal'
    os.makedirs(simu_dir, exist_ok=True)
    simulationReal(MPCStep, ADP_dir, simu_dir)

    # 5. 虚拟时域ADP、MPC应用
    simu_dir = ADP_dir + '/simulationVirtual'
    os.makedirs(simu_dir, exist_ok=True)

    for seed in range(100):
        print('seed={}'.format(seed))
        simulationVirtual(MPCStep, ADP_dir, simu_dir, noise = 0.5, seed = seed)

    errorList = np.loadtxt(simu_dir + "/RelError.csv", delimiter=',', skiprows=1)
    print('Mean Acceleration Error | Mean: {:.4f}%, Max: {:.4f}%'.format(np.mean(errorList[:,0])*100,np.mean(errorList[:,1])*100))
    print('Mean Steering Angle Error | Mean: {:.4f}%, Max: {:.4f}%'.format(np.mean(errorList[:,2])*100,np.mean(errorList[:,3])*100))
    print('Mean Distance Error Error | Mean: {:.4f}%, Max: {:.4f}%'.format(np.mean(errorList[:,4])*100,np.mean(errorList[:,5])*100))
    print('Mean Heading Angle Error Error | Mean: {:.4f}%, Max: {:.4f}%'.format(np.mean(errorList[:,6])*100,np.mean(errorList[:,7])*100))
    print('Mean Utility  Function Error | Mean: {:.4f}%, Max: {:.4f}%'.format(np.mean(errorList[:,8])*100,np.mean(errorList[:,9])*100))

if __name__ == '__main__':
    ADP_dir = './Results_dir/2022-04-21-15-12-31'
    main(ADP_dir)