# from matplotlib import pyplot as plt
# import torch
# import numpy as np
# from myenv import TrackingEnv

# refState = [1,2,3]
# print(torch.tensor([refState])[0].tolist())
# LifeMax = 40
# batchSize = 1000
# statelifeMax = torch.rand(batchSize) * LifeMax
# batchDataLife = torch.zeros(batchSize)

# for i in range(1000000000):
#     batchDataLife += 1
#     temp = (batchDataLife > statelifeMax)
#     if sum(temp) > 0:
#         batchDataLife[temp] = 0
#         statelifeMax[temp] = torch.rand(sum(temp)) * LifeMax

#     plt.figure()
#     plt.hist(batchDataLife.tolist(), bins=40, weights = np.zeros_like(batchDataLife.tolist()) + 1 / len(batchDataLife.tolist()))
#     plt.xlabel('Life')
#     plt.ylabel('Frequency')
#     plt.title('i='+str(i))
#     plt.savefig('./Simulation_dir/life_distribution.png')
#     plt.close()
a = None
b = 1
print(a==None)
print(b==None)
    

