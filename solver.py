from myenv import TrackingEnv
import matplotlib.pyplot as plt
from config import MPCConfig
from sys import path
# path.append(r"F:/casadi-windows-py36-v3.5.5-64bit")
from casadi import *

class Solver():
    def __init__(self):
        self._sol_dic = {'ipopt.print_level': 0,
                         'ipopt.sb': 'yes', 'print_time': 0}
        self.env = TrackingEnv()
        self.stateLow = self.env.stateLow[3:] + self.env.stateLow[:3]
        self.stateHigh = self.env.stateHigh[3:] + self.env.stateHigh[:3]
        self.actionLow = self.env.actionLow
        self.actionHigh = self.env.actionHigh
        self.actionDim = self.env.actionSpace.shape[0]
        self.stateDim = 6
        config = MPCConfig()
        self.gammar = config.gammar
        self.T = self.env.T
        state = SX.sym('state', self.stateDim)
        action = SX.sym('action', self.actionDim)
        # 替换model
        self.T = 0.1  # 时间间隔
        self.m = 1520  # 自车质量
        self.a = 1.19  # 质心到前轴的距离
        self.b = 1.46  # 质心到后轴的距离
        self.kf = -155495  # 前轮总侧偏刚度
        self.kr = -155495  # 后轮总侧偏刚度
        self.Iz = 2642  # 转动惯量
        stateNextt = vertcat(
            state[0] + self.T * (state[3] * cos(state[2]) - state[4] * sin(state[2])),
            state[1] + self.T * (state[4] * cos(state[2]) + state[3] * sin(state[2])),
            state[2] + self.T * state[5],
            state[3] + self.T * action[0],
            (-(self.a * self.kf - self.b * self.kr) * state[5] + self.kf * action[1] * state[3] +
                self.m * state[5] * state[3] * state[3] - self.m * state[3] * state[4] / self.T) \
                / (self.kf + self.kr - self.m * state[3] / self.T),
            (-self.Iz * state[5] * state[3] / self.T - (self.a * self.kf - self.b * self.kr) * state[4]
                    + self.a * self.kf * action[1] * state[3]) \
                / ((self.a * self.a * self.kf + self.b * self.b * self.kr) - self.Iz * state[3] / self.T)
        )
        self.F = Function("F", [state, action], [stateNextt])

        refState = SX.sym('refState',3 * self.env.refNum)
        cost = pow(state[0] - refState[0], 2) +\
            4 * pow(state[1] - refState[1], 2) +\
            10 * pow(state[2] - refState[2], 2) +\
            pow(action[0], 2) +\
            0.2 * pow(action[1], 2) # 
        self.calCost = Function('calCost', [state, refState, action], [cost])

    def MPCSolver(self, initState, refState, predictStep, isReal = True):
        # x: optimization variable
        # g: inequality constraints
        # J: cost function
        x = []
        lbx = []
        ubx = []
        lbg = []
        ubg = []
        G = []
        J = 0
        Xk = SX.sym('X0', self.stateDim)
        x += [Xk]
        lbx += initState
        ubx += initState
        gammar = 1
        for k in range(1, predictStep + 1):
            Uname = 'U' + str(k-1)
            Uk = SX.sym(Uname, self.actionDim)

            # add control to optimization variable
            x += [Uk]
            lbx += self.actionLow
            ubx += self.actionHigh

            # cost function
            J += self.calCost(Xk, refState, Uk) * gammar
            gammar *= self.gammar
            ######################### Real/vVirtual reference points update ############################
            if isReal==True:
                refState = self.env.refDynamicReal(refState, MPCflag=1)
            else:
                refState = self.env.refDynamicVirtual(refState, MPCflag=1)

            # Dynamic Constraints
            XNext = self.F(Xk, Uk)
            Xname = 'X' + str(k)
            Xk = SX.sym(Xname, self.stateDim)
            G += [XNext - Xk]
            lbg += [0 for _ in range(self.stateDim)]
            ubg += [0 for _ in range(self.stateDim)]

            # add state to optimization variable
            x += [Xk]
            lbx += self.stateLow
            ubx += self.stateHigh
        nlp = dict(f=J, g=vertcat(*G), x=vertcat(*x))
        solver = nlpsol('res', 'ipopt', nlp, self._sol_dic)
        # print(solver)
        res = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=0)
        # save result
        resX = np.array(res['x'])
        # print(res['x'])
        resState = np.zeros([predictStep, self.stateDim], dtype='float32')
        resControl = np.zeros([predictStep, self.actionDim], dtype='float32')
        totalDim = self.stateDim + self.actionDim
        for i in range(predictStep):
            resState[i] = resX[totalDim * i: totalDim * i + self.stateDim].reshape(-1)
            resControl[i] = resX[totalDim * i + self.stateDim: totalDim * (i + 1)].reshape(-1)
        return resState, resControl
