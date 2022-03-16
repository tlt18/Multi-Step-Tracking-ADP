from config import MPCConfig
import matplotlib.pyplot as plt
from myenv import TrackingEnv
from datetime import datetime
import os
from solver import Solver
import numpy as np
import torch
from network import Actor, Critic
import time

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
        x = torch.linspace(1, 30*np.pi, 1000)
        y = env.referenceCurve(x)
        plt.xlim(-5, 100)
        plt.ylim(-1.1, 1.1)
        plt.plot(x, y, color='gray')
        controlMPC = np.empty((0,2))
        while(count<env.testStep):
            refState = env.referencePoint(state[0], MPCflag=1)
            _, control = solver.MPCSolver(state, refState, mpcstep)
            action = control[0].tolist()
            state = env.vehicleDynamic(state[0], state[1], state[2], state[3], state[4], state[5], action[0], action[1], MPCflag=1)
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
        np.savetxt(os.path.join(simu_dir, "controlFullMPC"+str(mpcstep)+".csv"), controlMPC, delimiter=',')

def simulationOpen(MPCStep, simu_dir):
    # MPC
    env = TrackingEnv()
    solver = Solver()
    for mpcstep in MPCStep:
        print("----------------------Start Solving!----------------------")
        print("MPCStep: {}".format(mpcstep))
        # plt.ion()
        plt.figure(mpcstep)
        state = env.initState
        count = 0
        x = torch.linspace(1, 30*np.pi, 1000)
        y = env.referenceCurve(x)
        plt.xlim(-5, 100)
        plt.ylim(-1.1, 1.1)
        plt.plot(x, y, color='gray')
        refState = env.referencePoint(state[0]+60, MPCflag=1)
        _, control = solver.MPCSolver(state, refState, mpcstep)
        while(count<mpcstep):
            action = control[count].tolist()
            state = env.vehicleDynamic(state[0], state[1], state[2], state[3], state[4], state[5], action[0], action[1], MPCflag=1)
            plt.scatter(state[0], state[1], color='red', s=5)
            plt.scatter(refState[0], refState[1], color='blue', s=5)
            # plt.pause(0.05)
            count += 1
            print('count/totalStep: {}/{}'.format(count, env.testStep))
        plt.title('MPCStep:'+str(mpcstep))
        plt.savefig(simu_dir + '/MPCStep'+str(mpcstep)+'_open.png')
        # plt.ioff()
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
    refState = env.calRefState(initialState)
    timeStart = time.time()
    controlADP = policy(refState).detach()
    timeADP = (time.time() - timeStart)
    controlADP = controlADP.numpy()
    np.savetxt(os.path.join(simu_dir, "controlOneStepADP.csv"), controlADP, delimiter=',')
    print("ADP consumes {:.3f}s {} step".format(timeADP, stateNum))
    for mpcstep in MPCStep:
        timeMPC = 0
        controlMPC = np.empty((0,2))
        print("----------------------Start Solving MPC"+str(mpcstep)+"!----------------------")
        for i in range(stateNum):
            state = initialState[i].tolist()
            timeStart = time.time()
            _, control = solver.MPCSolver(state[0:6], state[6:], mpcstep)
            timeMPC += time.time() - timeStart
            controlMPC = np.append(controlMPC, control[0])
        timeMPC = timeMPC
        controlMPC = np.reshape(controlMPC, (-1, actionDim))
        print("MPC{} consumes {:.3f}s {} step".format(mpcstep, timeMPC, stateNum))
        np.savetxt(os.path.join(simu_dir, "controlOneStepMPC"+str(mpcstep)+".csv"), controlMPC, delimiter=',')
        # maxAction = np.max(controlMPC, 0)
        # minACtion = np.min(controlMPC, 0)
        maxAction = np.array(env.actionHigh)
        minAction = np.array(env.actionLow)
        relativeError = np.abs((controlADP - controlMPC)/(maxAction - minAction))
        relativeErrorMax = np.max(relativeError, 0)
        for i in range(actionDim):
            print('max relative error for action{}: {:.1f}%'.format(i+1, relativeErrorMax[i]*100))

def simulationMPC(MPCStep, simu_dir):
    # MPC
    env = TrackingEnv()
    env.seed(0)
    stateDim = env.stateDim - 2
    actionDim = env.actionSpace.shape[0]
    solver = Solver()
    for mpcstep in MPCStep:
        print("----------------------Start Solving!----------------------")
        print("MPCStep: {}".format(mpcstep))
        # plt.ion()
        plt.figure(mpcstep)
        state = env.initState[0]
        count = 0
        x = torch.linspace(1, 30*np.pi, 1000)
        y = env.referenceCurve(x)
        plt.xlim(-5, 100)
        plt.ylim(-1.1, 1.1)
        plt.plot(x, y, color='gray')
        controlMPC = np.empty((0,2))
        refState = env.referencePoint(state[0], MPCflag=1)
        while(count<env.testStep):
            # refState = env.referencePoint(state[0], MPCflag=1)
            _, control = solver.MPCSolver(state, refState, mpcstep)
            action = control[0].tolist()
            state = env.vehicleDynamic(state[0], state[1], state[2], state[3], state[4], state[5], action[0], action[1], MPCflag=1)
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
        np.savetxt(os.path.join(simu_dir, "controlOnlyMPC"+str(mpcstep)+".csv"), controlMPC, delimiter=',')


if __name__ == '__main__':
    config = MPCConfig()
    MPCStep = config.MPCStep

    # simu_dir = "./Simulation_dir/" + datetime.now().strftime("%Y-%m-%d-%H-%M")
    # os.makedirs(simu_dir, exist_ok=True)
    # # 真实时域中MPC表现（参考点更新方式和ADP一致）
    # simulationMPC(MPCStep, simu_dir)

    ## MPC单点跟踪的开环控制
    # simulationOpen(MPCStep, simu_dir)
    ADP_dir = './Results_dir/2022-03-16-17-14-53'
    simu_dir = ADP_dir + '/' + datetime.now().strftime("%Y-%m-%d-%H-%M")
    os.makedirs(simu_dir, exist_ok=True)
    simulationOneStep(MPCStep, ADP_dir, simu_dir, stateNum=200)
    simulationFull(MPCStep, ADP_dir, simu_dir)
