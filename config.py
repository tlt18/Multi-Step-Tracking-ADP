class trainConfig():
    def __init__(self):
        self.iterationMax = 50000
        self.iterationPrint = 10
        self.iterationSave = 500
        self.lrPolicy = 6e-4
        self.lrValue = 6e-3
        self.stepForwardPEV = 5
        self.stepForwardPIM = 3
        self.batchSize = 256
