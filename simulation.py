from cProfile import label
from curses.ascii import isprint
import math
import os
import time
from datetime import datetime
from turtle import color

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import MPCConfig
from myenv import TrackingEnv
from network import Actor, Critic
from solver import Solver

def simulationReal(MPCStep, ADP_dir, simu_dir, refNum = None, curveType = 'sine', seed = 0):
    print("----------------------Curve Type: {}----------------------".format(curveType))
    plotDelete = 0
    # 真实时域ADP、MPC应用
    env = TrackingEnv()
    env.seed(seed)
    if refNum != None:
        env.changeRefNum(refNum)
    relstateDim = env.relstateDim
    actionDim = env.actionSpace.shape[0]
    policy = Actor(relstateDim, actionDim)
    policy.loadParameters(ADP_dir)
    value = Critic(relstateDim, 1)
    value.loadParameters(ADP_dir)
    solver = Solver(env)
    # refIDinit
    if curveType == 'sine':
        initialState, info = env.resetSpecific(1, noise = -1, refIDinit = 0) # [u,v,omega,[xr,yr,phir],x,y,phi]
    elif curveType == 'DLC':
        initialState, info = env.resetSpecific(1, noise = -1, refIDinit = 1)
    elif curveType == 'Circle':
        initialState, info = env.resetSpecific(1, noise = -1, refIDinit = 2)
    # ADP
    stateAdp = initialState.clone()
    infoAdp = info.clone()
    controlADPList = np.empty(0)
    stateADPList = np.empty(0)
    rewardADP = np.empty(0)
    timeADP = np.empty(0)
    count = 0
    while(count < env.testStepReal[curveType]):
        stateADPList = np.append(stateADPList, stateAdp[0, -3:].numpy()) # x, y, phi
        stateADPList = np.append(stateADPList, stateAdp[0, :-3].numpy()) # u, v, omega, [xr, yr, phir]
        relState = env.relStateCal(stateAdp)
        start = time.time()
        controlAdp = policy(relState)
        end = time.time()
        timeADP = np.append(timeADP, end - start)
        controlAdp = controlAdp.detach()
        stateAdp, reward, done, infoAdp = env.stepSpecificRef(stateAdp, controlAdp, infoAdp)
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
    timeMPCAll = []
    for mpcstep in MPCStep:
        env.randomTestReset()
        print("Start Solving MPC-{}!".format(mpcstep))
        tempstate = initialState[0].tolist()
        infoMpc = info[0].tolist()
        stateMpc = tempstate[-3:] + tempstate[:3] # x, y, phi, u, v, omega
        refStateMpc = tempstate[3:-3]
        count = 0
        controlMPCList = np.empty(0)
        stateMPCList = np.empty(0)
        rewardMPC = np.empty(0)
        timeMPC = np.empty(0)
        while(count < env.testStepReal[curveType]):
            # MPC
            start = time.time()
            _, control = solver.MPCSolver(stateMpc, refStateMpc, mpcstep, isReal = True, info = infoMpc)
            end = time.time()
            timeMPC = np.append(timeMPC, end - start)
            stateMPCList = np.append(stateMPCList, np.array(stateMpc))
            stateMPCList = np.append(stateMPCList, np.array(refStateMpc))
            action = control[0].tolist()
            reward = env.calReward(stateMpc[-3:] + refStateMpc + stateMpc[:3],action,MPCflag=1)
            stateMpc = env.vehicleDynamic(
                stateMpc[0], stateMpc[1], stateMpc[2], stateMpc[3], stateMpc[4], stateMpc[5], action[0], action[1], MPCflag=1)
            refStateMpc[:-3] = refStateMpc[3:]
            refStateMpc[-3:] = [
                env.trajectoryList.calx(infoMpc[0] + env.refNum * env.T, infoMpc[1], MPCflag = 1),
                env.trajectoryList.caly(infoMpc[0]  + env.refNum * env.T, infoMpc[1], MPCflag = 1),
                env.trajectoryList.calphi(infoMpc[0]  + env.refNum * env.T, infoMpc[1], MPCflag = 1),
            ]
            infoMpc[0] += env.T
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
        timeMPCAll.append(timeMPC)
    
    print("Time consume ADP: {}ms".format(timeADP.mean() * 1000))
    for i in range(len(MPCStep)):
        print("Time consume MPC-{}: {}ms".format(MPCStep[i], timeMPCAll[i].mean() * 1000))

    colorList = ['darkorange', 'green', 'blue', 'red']
    plt.figure()
    pos = list(range(len(MPCStep) + 1))
    plt.bar([p for p in pos], [timeADP.mean() * 1000]+[timempc.mean() * 1000 for timempc in timeMPCAll], 
        width = 0.3,color = [colorList[-1]] + [colorList[i] for i in range(len(MPCStep))], 
        label=['ADP'] + ['MPC-'+str(mpcstep) for mpcstep in MPCStep])
    plt.xticks(range(len(MPCStep) + 1), ['ADP'] + ['MPC-'+str(mpcstep) for mpcstep in MPCStep])
    for x,y in enumerate([timeADP.mean() * 1000]+[timempc.mean() * 1000 for timempc in timeMPCAll]):
        plt.text(x, y,'%s ms' %round(y, 2), ha='center', va='bottom',fontsize=9)
    # plt.bar(['ADP'] + ['MPC-'+str(mpcstep) for mpcstep in MPCStep], 
    #     [timeADP.mean() * 1000]+[timempc.mean() * 1000 for timempc in timeMPCAll],
    #     color = [colorList[-1]] + [colorList[i] for i in range(len(MPCStep))],
    #     width=0.3)
    plt.ylabel("Average calculation time [ms]")
    plt.yscale('log')
    plt.savefig(simu_dir + '/average-calculation-time.png', bbox_inches='tight')
    # plt.title("Calculation time")
    plt.close()

    plt.figure()
    for i in range(len(MPCStep)):
        plt.plot(range(len(timeMPCAll[i])), timeMPCAll[i] * 1000, label = 'MPC-' + str(MPCStep[i]), color = colorList[i])
    plt.plot(range(len(timeADP)), timeADP * 1000, label = 'ADP', color = colorList[-1])
    plt.legend()
    plt.ylabel("Calculation time [ms]")
    plt.xlabel('Step')
    plt.yscale('log')
    # plt.title("Calculation time")
    plt.savefig(simu_dir + '/calculation-time-step.png', bbox_inches='tight')
    plt.close()

    # TODO: time
    plt.figure()
    plt.boxplot([time * 1000 for time in timeMPCAll] + [timeADP * 1000], patch_artist=True,widths=0.4,
                showmeans=True,
                meanprops={'marker':'+',
                        'markerfacecolor':'k',
                        'markeredgecolor':'k',
                        'markersize':5})
    plt.xticks(range(1, len(MPCStep)+2, 1), ['MPC-'+str(step) for step in MPCStep] + ['RL'])
    # plt.ylim(0,9)
    plt.yscale('log')
    plt.grid(axis='y',ls='--',alpha=0.5)
    plt.ylabel('Calculation time [ms]',fontsize=18)
    plt.savefig(simu_dir + '/boxplot-time.png', bbox_inches='tight')
    plt.close()

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
    figSize = (20,5)
    # y v.s. x
    xADP = stateADPList[:,0]
    xMPC = [mpc[:,0] for mpc in stateMPCAll]
    xRef = stateADPList[:,6]
    yADP = stateADPList[:,1]
    yMPC = [mpc[:,1] for mpc in stateMPCAll]
    yRef = stateADPList[:,7]
    xName = 'X [m]'
    yName = 'Y [m]'
    title = 'y-x'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isRef = True, xRef = xRef, yRef = yRef, figSize='equal', lineWidth = 2)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isRef = True, xRef = xRef, yRef = yRef, lineWidth = 2)

    # distance error v.s. t
    yADP = np.sqrt(np.power(stateADPList[:, 0] - stateADPList[:, 6], 2) + np.power(stateADPList[:, 1] - stateADPList[:, 7], 2))*100
    yMPC = [np.sqrt(np.power(mpc[:, 0] - mpc[:, 6], 2) + np.power(mpc[:, 1] - mpc[:, 7], 2))*100 for mpc in stateMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'Time [s]'
    yName = 'Distance error [cm]'
    title = 'distance-error-t'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, figSize=figSize)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title)
    Ip_ADP = np.sqrt(np.mean(np.power(stateADPList[:, 0] - stateADPList[:, 6], 2) + np.power(stateADPList[:, 1] - stateADPList[:, 7], 2)))
    Ip_MPC = [np.sqrt(np.mean(np.power(mpc[:, 0] - mpc[:, 6], 2) + np.power(mpc[:, 1] - mpc[:, 7], 2))) for mpc in stateMPCAll]
    print('Position error ADP: {}m'.format(Ip_ADP))
    for i in range(len(MPCStep)):
        print('Position error MPC-{}: {}m'.format(MPCStep[i], Ip_MPC[i]))

    # x error v.s. t
    yADP = stateADPList[:, 0] - stateADPList[:, 6]
    yMPC = [mpc[:, 0] - mpc[:, 6] for mpc in stateMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'time [s]'
    yName = 'X error [m]'
    title = 'x-error-t'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, figSize=figSize)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title)

    # y error v.s. t
    yADP = stateADPList[:, 1] - stateADPList[:, 7]
    yMPC = [mpc[:, 1] - mpc[:, 7] for mpc in stateMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'time [s]'
    yName = 'Y error [m]'
    title = 'y-error-t'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, figSize=figSize)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title)

    # phi v.s. t
    yADP = stateADPList[:,2] * 180/np.pi
    yMPC = [mpc[:,2] * 180/np.pi for mpc in stateMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'time [s]'
    yName = 'Heading angle [°]'
    title = 'phi-t'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, figSize=figSize)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title)

    # phi error v.s. t
    yADP = stateADPList[:,2] * 180/np.pi - stateADPList[:,8] * 180/np.pi
    yMPC = [mpc[:,2] * 180/np.pi - mpc[:,8] * 180/np.pi for mpc in stateMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'time [s]'
    yName = 'Heading angle error [°]'
    title = 'phi-error-t'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, figSize=figSize)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title)
    Iphi_ADP = np.sqrt(np.mean(np.power(stateADPList[:,2] * 180/np.pi - stateADPList[:,8] * 180/np.pi, 2)))
    Iphi_MPC = [np.sqrt(np.mean(np.power(mpc[:,2] * 180/np.pi - mpc[:,8] * 180/np.pi, 2))) for mpc in stateMPCAll]
    print('Phi error ADP: {}°'.format(Iphi_ADP))
    for i in range(len(MPCStep)):
        print('Phi error MPC-{}: {}°'.format(MPCStep[i], Iphi_MPC[i]))

    # utility v.s. t
    yADP = rewardADP
    yMPC = [mpc for mpc in rewardMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'time [s]'
    yName = 'Utility'
    title = 'utility-t'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, figSize=figSize)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title)

    # accumulated utility v.s. t
    yADP = np.cumsum(rewardADP)
    yMPC = [np.cumsum(mpc) for mpc in rewardMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'time [s]'
    yName = 'Accumulated utility'
    title = 'accumulated-utility-t'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, figSize=figSize)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title)
    print('Accumulated utility of ADP {:.4f}, {:.4f}% higher than MPC'.format(yADP[-1], (yADP[-1]-yMPC[-1][-1])/yMPC[-1][-1]*100))
    for i in range(len(yMPC)):
        print('Accumulated utility of MPC-{} {:.4f}, {:.4f}% higher than MPC-{}'.format(
            MPCStep[i], yMPC[i][-1], (yMPC[i][-1]-yMPC[-1][-1])/yMPC[-1][-1]*100, MPCStep[-1]))
    # a v.s. t
    yADP = controlADPList[:,0]
    yMPC = [mpc[:,0] for mpc in controlMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'time [s]'
    yName = 'a [m/s^2]'
    title = 'a-t'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, figSize=figSize)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title)

    # delta v.s. t
    yADP = controlADPList[:,1] * 180/np.pi
    yMPC = [mpc[:,1] * 180/np.pi for mpc in controlMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'time [s]'
    yName = 'delta [°]'
    title = 'delta-t'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, figSize=figSize)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title)
 
def simulationVirtual(MPCStep, ADP_dir, simu_dir, noise = 0, seed = 0):
    # 虚拟时域ADP、MPC应用
    print("----------------------Start Solving! seed: {}----------------------".format(seed))
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
    initialState = env.resetRandom(1, noise = noise) # [u,v,omega,[xr,yr,phir],x,y,phi]
    if noise == 0:
        # initialState[0, 0] += 0.01
        initialState[0, 2] = -0.05
        initialState[0, -2] = 0.8
        initialState[0, -1] = np.pi/30

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
        with open(simu_dir + "/simulationVirtualMPC_"+str(mpcstep)+".csv", 'wb') as f:
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
    xADP = np.arange(0, len(controlADPList[:,0])) * env.T
    xMPC = [np.arange(0, len(mpc[:,0])) * env.T for mpc in controlMPCAll]
    yADP = controlADPList[:,0]
    yMPC = [mpc[:,0] for mpc in controlMPCAll]
    xName = 'Time [s]'
    yName = 'Acceleration [m/s^2]'
    title = 'a-t'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isError = True)

    # delta v.s. t
    xADP = np.arange(0, len(controlADPList[:,0])) * env.T
    xMPC = [np.arange(0, len(mpc[:,1])) * env.T for mpc in controlMPCAll]
    yADP = controlADPList[:,1] * 180 / np.pi
    yMPC = [mpc[:,1] * 180 / np.pi for mpc in controlMPCAll]
    xName = 'Time [s]'
    yName = 'Steering Angle [°]'
    title = 'delta-t'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isError = True)

    # distance error v.s. t
    yADP = np.sqrt(np.power(stateADPList[:, 0] - stateADPList[:, 6], 2) + np.power(stateADPList[:, 1] - stateADPList[:, 7], 2))
    yMPC = [np.sqrt(np.power(mpc[:, 0] - mpc[:, 6], 2) + np.power(mpc[:, 1] - mpc[:, 7], 2)) for mpc in stateMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'Time [s]'
    yName = 'Distance error [m]'
    title = 'distance-error-t'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isError = True)

    # phi error v.s. t
    yADP = (stateADPList[:,2] - stateADPList[:,8]) * 180 / np.pi
    yMPC = [(mpc[:,2] - mpc[:,8]) * 180 / np.pi for mpc in stateMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'Time [s]'
    yName = 'Heading angle error [°]'
    title = 'phi-error-t'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isError = True)

    # phi v.s. t
    yADP = stateADPList[:,2] * 180 / np.pi
    yMPC = [mpc[:,2] * 180 / np.pi for mpc in stateMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'Time [s]'
    yName = 'Heading angle [°]'
    title = 'phi-t'
    xRef = np.arange(0, len(yADP)) * env.T
    yRef = stateADPList[:,8] * 180 / np.pi
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isRef = True, xRef = xRef, yRef = yRef)

    # phi error v.s. distance error
    xADP = np.sqrt(np.power(stateADPList[:, 0] - stateADPList[:, 6], 2) + np.power(stateADPList[:, 1] - stateADPList[:, 7], 2))
    xMPC = [np.sqrt(np.power(mpc[:, 0] - mpc[:, 6], 2) + np.power(mpc[:, 1] - mpc[:, 7], 2)) for mpc in stateMPCAll]
    yADP = (stateADPList[:,2] - stateADPList[:,8]) * 180 / np.pi
    yMPC = [(mpc[:,2] - mpc[:,8]) * 180 / np.pi for mpc in stateMPCAll]
    xName = 'Distance error [m]'
    yName = 'Heading angle error [°]'
    title = 'phi-error-Distance-error'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isError = False)

    # y v.s. x
    xADP = stateADPList[:, 0]
    xMPC = [mpc[:, 0] for mpc in stateMPCAll]
    yADP = stateADPList[:, 1]
    yMPC = [mpc[:, 1] for mpc in stateMPCAll]
    xName = 'X [m]'
    yName = 'Y [m]'
    title = 'y-x'
    xRef = stateADPList[:, 6]
    yRef = stateADPList[:, 7]
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isRef = True, xRef = xRef, yRef = yRef)

    # utility v.s. t
    yADP = rewardADP
    yMPC = [mpc for mpc in rewardMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'Time [s]'
    yName = 'utility'
    title = 'utility-t'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isError = True)

    # accumulated utility v.s. t
    yADP = np.cumsum(rewardADP)
    yMPC = [np.cumsum(mpc) for mpc in rewardMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'Time [s]'
    yName = 'Accumulated cost'
    title = 'accumulated-cost-t'
    comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = True, isError = False)

def comparePlot(xADP, xMPC, yADP, yMPC, MPCStep, xName, yName, simu_dir, title, isMark = False, isError = False, isRef = False, xRef = None, yRef = None, figSize = None, lineWidth = 2):
    if figSize != None and figSize != 'equal':
        plt.figure(figsize=figSize, dpi=300)
    else:
        plt.figure()
    colorList = ['darkorange', 'limegreen', 'blue', 'red']
    if isMark == True:
        markerList = ['|', 'D', 'o', '*']
    else:
        markerList = ['None', 'None', 'None', 'None']
    for i in range(len(xMPC)):
        plt.plot(xMPC[i], yMPC[i], linewidth=lineWidth, color = colorList[3 - len(xMPC) + i], linestyle = '--', marker=markerList[3 - len(xMPC) + i], markersize=4)

    plt.plot(xADP, yADP , linewidth = lineWidth, color=colorList[-1],linestyle = '--', marker=markerList[-1], markersize=4)
    if isError == True:
        plt.plot([np.min(xADP), np.max(xADP)], [0,0], linewidth = lineWidth/2, color = 'grey', linestyle = '--')
        plt.legend(labels=['MPC'+str(mpcStep) for mpcStep in MPCStep] + ['ADP', 'Ref'])
    elif isRef == True:
        plt.plot(xRef, yRef, linewidth = lineWidth/2, color = 'gray', linestyle = '--')
        plt.legend(labels=['MPC'+str(mpcStep) for mpcStep in MPCStep] + ['ADP', 'Ref'])
        # plt.legend(labels=['MPC'+str(mpcStep) for mpcStep in MPCStep] + ['ADP', 'Ref'])
    else:
        plt.legend(labels=['MPC'+str(mpcStep) for mpcStep in MPCStep] + ['ADP'])
    plt.xlabel(xName)
    plt.ylabel(yName)
    # plt.savefig(simu_dir + '/' + title + '.png', bbox_inches='tight')
    plt.savefig(simu_dir + '/' + title + '.png')
    if figSize == 'equal':
        plt.axis('equal')
    else:
        plt.axis('scaled')
    plt.close()

def comparePlotADP(xADP, yADP, refNum_list, xName, yName, simu_dir, title, isRef = False, xRef = None, yRef = None):
    if isRef == True:
        plt.plot(xRef, yRef, linewidth = 2, color = 'black', label = 'ref')
    for i in range(len(xADP)):
        plt.plot(xADP[i], yADP[i], linewidth = 2, label = 'ADP(N=' + str(refNum_list[i]) + ')')
    plt.legend()
    plt.xlabel(xName)
    plt.ylabel(yName)
    # plt.savefig(simu_dir + '/' + title + '.png', bbox_inches='tight')
    plt.savefig(simu_dir + '/' + title + '.png')
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

def calRelError(ADP, MPC, title, simu_dir, isPlot = False, isPrint = True):
    maxMPC = np.max(MPC, 0)
    minMPC = np.min(MPC, 0)
    relativeError = np.abs((ADP - MPC)/(maxMPC - minMPC + 1e-3))
    relativeErrorMax = np.max(relativeError, 0)
    relativeErrorMean = np.mean(relativeError, 0)
    if isPrint == True:
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

def simuVirtualTraning(env, ADP_dir, noise = -1, refIDinit = 0):
    config = MPCConfig()
    mpcstep = max(config.MPCStep)
    relstateDim = env.relstateDim
    actionDim = env.actionSpace.shape[0]
    # ADP
    policy = Actor(relstateDim, actionDim)
    policy.loadParameters(ADP_dir)
    count = 0
    stateAdp, infoADP = env.resetSpecific(env.testSampleNum, noise = noise, refIDinit = refIDinit)
    controlADPList = np.empty(0)
    rewardList = np.empty(0)
    if refIDinit == 0:
        testStep = env.testStepReal["sine"]
    elif refIDinit == 1:
        testStep = env.testStepReal["DLC"]
    while(count < testStep):
        relState = env.relStateCal(stateAdp)
        controlAdp = policy(relState).detach()
        stateAdp, reward, done, infoADP = env.stepSpecificRef(stateAdp, controlAdp, infoADP)
        controlADPList = np.append(controlADPList, controlAdp.numpy())
        rewardList = np.append(rewardList, reward.numpy().mean())
        count += 1
    controlADPList =np.reshape(controlADPList, (testStep, env.testSampleNum, actionDim))
    ADPAction = np.array(np.transpose(controlADPList, (1, 0, 2)))

    return rewardList.mean()

def main(ADP_dir, RefNum):
    config = MPCConfig()
    MPCStep = config.MPCStep

    parameters = {'axes.labelsize': 20,
        'axes.titlesize': 18,
    #   'figure.figsize': (9.0, 6.5),
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'axes.unicode_minus': False,
        'font.size': 12.5,
        'figure.figsize': (10, 6.4)
        }
    plt.rcParams.update(parameters)

    simu_dir = ADP_dir + '/simulationReal/sine'
    os.makedirs(simu_dir, exist_ok=True)
    simulationReal(MPCStep, ADP_dir, simu_dir, refNum = refNum, curveType = 'sine')

    simu_dir = ADP_dir + '/simulationReal/DLC'
    os.makedirs(simu_dir, exist_ok=True)
    simulationReal(MPCStep, ADP_dir, simu_dir, refNum = refNum, curveType = 'DLC')

    simu_dir = ADP_dir + '/simulationReal/Circle'
    os.makedirs(simu_dir, exist_ok=True)
    simulationReal(MPCStep, ADP_dir, simu_dir, refNum = refNum, curveType = 'Circle')

def compareHorizon(ADP_list, refNum_list, curveType = 'sine', seed = 0):
    simu_dir = "./Simulation_dir/compareHorizon" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(simu_dir, exist_ok=True)
    print("----------------------Curve Type: {}----------------------".format(curveType))
    plotDelete = 0
    env = TrackingEnv()
    env.seed(seed)
    controlAll = []
    stateAll = []
    rewardAll = []
    for ADP_dir, refNum in zip(ADP_list, refNum_list):
        env.changeRefNum(refNum)
        relstateDim = env.relstateDim
        actionDim = env.actionSpace.shape[0]
        policy = Actor(relstateDim, actionDim)
        policy.loadParameters(ADP_dir)
        value = Critic(relstateDim, 1)
        value.loadParameters(ADP_dir)
        # refIDinit
        if curveType == 'sine':
            initialState, info = env.resetSpecific(1, noise = -1, refIDinit = 0) # [u,v,omega,[xr,yr,phir],x,y,phi]
        elif curveType == 'DLC':
            initialState, info = env.resetSpecific(1, noise = -1, refIDinit = 1)
        elif curveType == 'Circle':
            initialState, info = env.resetSpecific(1, noise = -1, refIDinit = 2)
        initialState[0, -2] += 0.2
        stateAdp = initialState.clone()
        infoAdp = info.clone()
        controlADPList = np.empty(0)
        stateADPList = np.empty(0)
        rewardADP = np.empty(0)
        timeADP = np.empty(0)
        count = 0
        while(count < env.testStepReal[curveType]):
            stateADPList = np.append(stateADPList, stateAdp[0, -3:].numpy()) # x, y, phi
            stateADPList = np.append(stateADPList, stateAdp[0, :-3].numpy()) # u, v, omega, [xr, yr, phir]
            relState = env.relStateCal(stateAdp)
            start = time.time()
            controlAdp = policy(relState)
            end = time.time()
            timeADP = np.append(timeADP, end - start)
            controlAdp = controlAdp.detach()
            stateAdp, reward, done, infoAdp = env.stepSpecificRef(stateAdp, controlAdp, infoAdp)
            controlADPList = np.append(controlADPList, controlAdp[0].numpy())
            rewardADP = np.append(rewardADP, reward.numpy())
            count += 1
        stateADPList = np.reshape(stateADPList, (-1, env.stateDim))
        controlADPList = np.reshape(controlADPList, (-1, actionDim))
        stateADPList = np.delete(stateADPList, range(plotDelete), 0)
        controlADPList = np.delete(controlADPList, range(plotDelete), 0)
        rewardADP = np.delete(rewardADP, range(plotDelete), 0)
        controlAll.append(controlADPList)
        stateAll.append(stateADPList)
        rewardAll.append(rewardADP)

    # Plot
    # stateAll: [x,y,phi,u,v,omega,[xr,yr,phir]]
    # controlAll: [a, delta]
    # y v.s. x
    xADP = [data[:,0] for data in stateAll]
    xRef = stateAll[0][:,6]
    yADP = [data[:,1] for data in stateAll]
    yRef = stateAll[0][:,7]
    xName = 'X [m]'
    yName = 'Y [m]'
    title = 'Y-X'
    comparePlotADP(xADP, yADP, refNum_list, xName, yName, simu_dir, title, isRef = True, xRef = xRef, yRef = yRef)

    # distance error v.s. t
    yADP = [np.sqrt(np.power(data[:, 0] - data[:, 6], 2) + np.power(data[:, 1] - data[:, 7], 2))*100 for data in stateAll]
    xADP = [np.arange(0, len(data)) * env.T for data in yADP]
    xName = 'Time [s]'
    yName = 'Distance error [cm]'
    title = 'distance-error-t'
    comparePlotADP(xADP, yADP, refNum_list, xName, yName, simu_dir, title, isRef = False)\

if __name__ == '__main__':
    # ADP_dir_list = [\
    #     './Results_dir/2023-02-13-16-26-22',\
    #     './Results_dir/2023-02-13-16-29-00',\
    #     './Results_dir/2023-02-13-16-29-28',\
    #     './Results_dir/2023-02-13-16-29-47',\
    #     './Results_dir/2023-02-14-12-15-23',\
    # ]
    # refNum_list = [1, 3, 5, 7, 9]
    # for ADP_dir, refNum in zip(ADP_dir_list, refNum_list):
    #     print('-' * 30 + 'refNum=' + str(refNum) + '-'*30)
    #     main(ADP_dir, refNum)

    # ADP_dir_list = [
    #     './Results_dir/2023-02-14-12-15-23'
    # ]
    # refNum_list = [9]
    # for ADP_dir, refNum in zip(ADP_dir_list, refNum_list):
    #     print('-' * 30 + 'refNum=' + str(refNum) + '-'*30)
    #     main(ADP_dir, refNum)
    

    parameters = {'axes.labelsize': 20,
        'axes.titlesize': 18,
    #   'figure.figsize': (9.0, 6.5),
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'axes.unicode_minus': False,
        'font.size': 12.5,
        'figure.figsize': (9, 6.4)
        }
    plt.rcParams.update(parameters)

    file_list = ['2023-02-13-16-26-22',\
        '2023-02-13-16-29-28',\
        '2023-02-14-12-15-23'
    ]
    ADP_list = ['./Results_dir/' + file for file in file_list]
    refNum_list = [1, 5, 9]

    compareHorizon(ADP_list, refNum_list, curveType = 'sine')
    compareHorizon(ADP_list, refNum_list, curveType = 'DLC')
    compareHorizon(ADP_list, refNum_list, curveType = 'Circle')