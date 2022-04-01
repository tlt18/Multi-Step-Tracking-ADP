from matplotlib import pyplot as plt
import torch
import numpy as np
from myenv import TrackingEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


