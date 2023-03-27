import os
import shutil
import time
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
import torch

from config import trainConfig, vehicleDynamicConfig
from myenv import TrackingEnv
from network import Actor, Critic
from train import Train
import simulation

os.environ["OMP_NUM_THREADS"] = "4"
torch.set_num_threads(4)

# mode setting
isTrain = True
isSimu = True

# parameters setting
config = trainConfig()
env = TrackingEnv()
env.seed(60)

use_gpu = torch.cuda.is_available()
relstateDim = env.relstateDim
actionDim = env.actionSpace.shape[0]
policy = Actor(relstateDim, actionDim, lr=config.lrPolicy)
value = Critic(relstateDim, 1, lr=config.lrValue)
# if use_gpu:
#     policy = policy.cuda()
#     value = value.cuda()

# ADP_dir = './Results_dir/2022-03-29-10-19-28'
# policy.loadParameters(ADP_dir)
# value.loadParameters(ADP_dir)
refNum = env.refNum
log_dir = "./Results_dir/refNum"+str(refNum)+'/'+ datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(log_dir + '/train', exist_ok=True)
os.makedirs(log_dir + '/code', exist_ok=True)

shutil.copy('./config.py', log_dir + '/code/config.py')
shutil.copy('./main.py', log_dir + '/code/main.py')
shutil.copy('./myenv.py', log_dir + '/code/myenv.py')
shutil.copy('./network.py', log_dir + '/code/network.py')
shutil.copy('./train.py', log_dir + '/code/train.py')
shutil.copy('./simulation.py', log_dir + '/code/simulation.py')
shutil.copy('./solver.py', log_dir + '/code/solver.py')
shutil.copy('./replaybuffer.py', log_dir + '/code/replaybuffer.py')
shutil.copy('./tensorboardplot.py', log_dir + '/code/tensorboardplot.py')


dataWriter = SummaryWriter(log_dir + '/train')

if isTrain: 
    print("----------------------Start Training!----------------------")
    train = Train(env, log_dir+'/train')
    iterarion = 0
    lossListValue = 0
    timeBegin = time.time()
    while iterarion < config.iterationMax:
        # PEV
        train.policyEvaluate(policy, value)
        # PIM
        train.policyImprove(policy, value)
        # update
        train.update(policy)
        # store loss
        dataWriter.add_scalar('Policy Loss', train.lossIteraPolicy.mean(), iterarion)
        dataWriter.add_scalar('Value Loss', train.lossIteraValue.mean(), iterarion)
        if iterarion % config.iterationSave == 0 or iterarion == config.iterationMax - 1:
            print("iteration: {}, LossValue: {:.4f}, LossPolicy: {:.4f}, value lr: {:10f}, policy lr: {:10f}".format(
                iterarion, train.lossIteraValue, train.lossIteraPolicy, value.opt.param_groups[0]['lr'], policy.opt.param_groups[0]['lr']))
            # save parameters
            value.saveParameters(log_dir)
            policy.saveParameters(log_dir)
            # test in real time
            env.policyTestReal(policy, iterarion, log_dir+'/train', curveType = 'sine')
            env.policyTestReal(policy, iterarion, log_dir+'/train', curveType = 'DLC')
            env.policyTestReal(policy, iterarion, log_dir+'/train', curveType = 'TurnLeft')
            env.policyTestReal(policy, iterarion, log_dir+'/train', curveType = 'TurnRight')
            env.policyTestReal(policy, iterarion, log_dir+'/train', curveType = 'RandomTest')
            # test in virtual time
            rewardSum1 = simulation.simuVirtualTraning(env, log_dir, noise = -1, refIDinit = 0)
            rewardSum2 = simulation.simuVirtualTraning(env, log_dir, noise = -1, refIDinit = 1)
            dataWriter.add_scalar('Sine cost', rewardSum1, iterarion)
            dataWriter.add_scalar('DLC cost', rewardSum2, iterarion)
            print("Accumulated Cost in sine is {:.4f}".format(rewardSum1))
            print("Accumulated Cost in DLC is {:.4f}".format(rewardSum2))

            # time consume
            timeDelta = time.time() - timeBegin
            h = timeDelta//3600
            mi = (timeDelta - h * 3600)//60
            sec = timeDelta % 60
            print("Time consuming: {:.0f}h {:.0f}min {:.0f}sec".format(h, mi, sec))
        iterarion += 1

if isSimu: 
    simulation.main(log_dir, refNum)
