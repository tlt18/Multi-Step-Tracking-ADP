import math
import numpy as np
import torch

class trainConfig():
    def __init__(self):
        self.iterationMax = 100000
        self.iterationPrint = 100
        self.iterationSave = 1000
        self.lrPolicy = 1e-3
        self.lrValue = 1e-2
        # self.lrPolicy = 1e-5
        # self.lrValue = 1e-4
        # need to match
        self.stepForwardPEV = 30
        self.gammar = 0.95
        self.refNoise = 1
        self.lifeMax = 30
        self.batchSize = 256
        self.sampleSize = 256
        self.warmBuffer = 4 * 256
        self.capacity = 256000


class vehicleDynamic():
    def __init__(self):
        self.refV = 5
        # y_r = curveA * sin(curveK * x_r)
        self.curveK = 1/6
        self.curveA = 1
        # Double lane chang
        self.DLCh = 3.5
        self.DLCa = 30
        self.DLCb = 50
        self.curvePhi = np.pi/60

        # 车辆参数
        # TODO: 参数是否合理？
        self.T = 0.1  # 时间间隔
        self.m = 1520  # 自车质量
        self.a = 1.19  # 质心到前轴的距离
        self.b = 1.46  # 质心到后轴的距离
        self.kf = -155495  # 前轮总侧偏刚度
        self.kr = -155495  # 后轮总侧偏刚度
        self.Iz = 2642  # 转动惯量

        # 初始状态
        self.initState = [0, 0, math.atan(self.curveA * self.curveK), self.refV, 0, 0]
        self.testStepReal = {'sine': 200, 'DLC': 350, 'TurnLeft': 30, 'TurnRight': 30, 'RandomTest': 500}
        self.testStepVirtual = 40
        # TODO: to 100
        self.testSampleNum = 10
        self.refNum = 10
        self.mpcstep = 60

class MPCConfig():
    def __init__(self):
        self.MPCStep = [10, 30, 60]
        config = trainConfig()
        self.gammar = config.gammar

