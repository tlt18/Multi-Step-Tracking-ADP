import math

class trainConfig():
    def __init__(self):
        self.iterationMax = 50000
        self.iterationPrint = 100
        self.iterationSave = 1000
        self.lrPolicy = 1e-3
        self.lrValue = 1e-2
        # self.lrPolicy = 1e-5
        # self.lrValue = 1e-4
        # need to match
        self.stepForwardPEV = 10
        self.gammar = 0.9

        self.lifeMax = 15
        self.batchSize = 256


class vehicleDynamic():
    def __init__(self):
        # 参考速度
        self.refV = 5
        self.curveK = 1/10
        self.curveA = 3
        # self.curveK = 1/5
        # self.curveA = 2

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

        self.testStepReal = 100
        self.testStepVirtual = 30
        self.testSampleNum = 60
        self.renderStep = 100
        self.refNum = 2

class MPCConfig():
    def __init__(self):
        self.MPCStep = [5, 10, 20]
        config = trainConfig()
        self.gammar = config.gammar
        # self.MPCStep = [205, 210, 220, 240]
        # self.MPCStep = [100]

