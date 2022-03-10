import myenv
import matplotlib.pyplot as plt
import config
from casadi import *
from sys import path
path.append(r"F:/casadi-windows-py36-v3.5.5-64bit")


class Solver():
    def __init__(self):
        self._sol_dic = {'ipopt.print_level': 0,
                         'ipopt.sb': 'yes', 'print_time': 0}
        self.env = TrackingEnv()
        self.stateLow = self.env.stateLow
        self.stateHigh = self.env.stateHigh
        self.actionLow = self.env.actionLow
        self.actionHigh = self.env.actionHigh
        self.actionDim = self.env.actionSpace.shape[0]
        self.stateDim = self.env.stateDim - 2
        state = SX.sym('state', self.stateDim)
        action = SX.sym('action', self.actionDim)
        stateNext = self.env.vehicleDynamic(
            state[0], state[1], state[2],
            state[3], state[4], state[5],
            action[0], action[1])
        stateNextt = vercat(stateNext[0], stateNext[1],
                            stateNext[2], stateNext[3], stateNext[4], stateNext[5])
        self.F = Function("F", [state, action], stateNextt)
        refState = SX.sym('refState', 2)
        cost = self.env.calReward(
            [state[0], state[1], state[2], state[3], state[4],
             state[5], refState[0], refState[1]], action)
        self.calCost = Function('calCost', [state, refstate, action], cost)

    def MPCSolver(self, initState, predictStep):
        # 计算参考点
        refx, refy = self.env.referencePoint(initState)
        # x: 优化变量
        # g: 不等式约束
        # J: 效用函数
        x = []
        lbx = []
        ubx = []
        lbg = []
        ubg = []
        G = []
        J = 0
        Xk = MX.sym('X0', self.stateDim)
        x += [Xk]
        lbx += initState
        ubx += initState
        for k in range(1, predictStep + 1):
            Uname = 'U' + str(k-1)
            Uk = MX.sym(Uname, self.actionDim)
            # 控制量加入优化变量
            x += [Uk]
            lbx += self.actionLow
            ubx += self.actionHigh
            # 代价函数
            J += self.calCost(Xk, [refx, refy], Uk)
            # 动力学约束
            XNext = self.F(Xk, Uk)
            Xname = 'X' + str(k)
            Xk = MX.sym(Xname, self.stateDim)
            G += [XNext - Xk]
            lbg += [0 for _ in range(self.stateDim)]
            ubg += [0 for _ in range(self.stateDim)]
            # 状态量加入优化变量
            x += [Xk]
            lbx += initState
            ubx += initState
        npl = dict(f=J, g=vertcat(*G), x=vertcat(*x))
        solver = nlpsol('res', 'iport', nlp, self._sol_dic)
        res = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=SX.zeros(2))
        # 保存结果
        resX = np.array(res['x'])
        resState = np.zeros([predictStep, self.stateDim])
        resControl = np.zeros([predictStep, self.actionDim])
        totalDim = self.stateDim + self.actionDim
        for i in range(predict_steps):
            state[i] = resX[nt * i: nt * i + self.stateDim].reshape(-1)
            control[i] = resX[nt * i + self.stateDim: nt * (i + 1)].reshape(-1)
        return state, control
