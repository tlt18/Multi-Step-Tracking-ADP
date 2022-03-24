import math

class trainConfig():
    def __init__(self):
        self.iterationMax = 50000
        self.iterationPrint = 10
        self.iterationSave = 500
        self.lrPolicy = 6e-4
        self.lrValue = 6e-3
        self.stepForwardPEV = 5
        self.stepForwardPIM = 4
        self.gammar = 0.8
        self.batchSize = 256
        self.updVirtual = True

class vehicleDynamic():
    def __init__(self):
        # 参考速度
        self.refV = 5
        self.curveK = 1/10
        self.curveA = 1
        # 固定参考点向前看个数
        self.refStep = 4
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

        self.testStep = 200
        self.renderStep = 100
        self.refNum = 1

class MPCConfig():
    def __init__(self):
        # self.MPCStep = [5, 10, 20, 30, 50, 100]
        self.MPCStep = [80]
        # self.MPCStep = [100]

