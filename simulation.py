from config import MPCConfig
import matplotlib.pyplot as plt
from myenv import TrackingEnv
from datetime import datetime
import os
from solver import Solver
import numpy as np
import torch

def simulation(MPCStep, simu_dir):
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
        # refState = env.referencePoint(state[0]+60, MPCflag=1)
        while(count<env.testStep):
            refState = env.referencePoint(state[0], MPCflag=1)
            _, control = solver.MPCSolver(state, refState, mpcstep)
            action = control[0].tolist()
            state = env.vehicleDynamic(state[0], state[1], state[2], state[3], state[4], state[5], action[0], action[1], MPCflag=1)
            plt.scatter(state[0], state[1], color='red', s=5)
            plt.scatter(refState[0], refState[1], color='blue', s=5)
            # plt.pause(0.05)
            count += 1
            print('count/totalStep: {}/{}'.format(count, env.testStep))
        plt.title('MPCStep:'+str(mpcstep))
        plt.savefig(simu_dir + '/MPCStep'+str(mpcstep)+'.png')
        # plt.ioff()
        plt.close()

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

if __name__ == '__main__':
    config = MPCConfig()
    MPCStep = config.MPCStep
    simu_dir = "./Simulation_dir/" + datetime.now().strftime("%Y-%m-%d-%H-%M")
    os.makedirs(simu_dir, exist_ok=True)
    # simulation(MPCStep, simu_dir)
    simulationOpen(MPCStep, simu_dir)



