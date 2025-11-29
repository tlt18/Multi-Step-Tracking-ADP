"""CasADi-based MPC solver used for the baselines and ablations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import casadi as ca
import l4casadi as l4c
import numpy as np

from .config import MPCConfig
from .env import TrackingEnv

__all__ = ["Solver"]


class Solver:
    """Thin wrapper around CasADi/IPOPT for solving the MPC problem."""

    def __init__(self, env: Optional[TrackingEnv] = None, value=None, value_multi=None):
        self._sol_dic = {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0}
        self.env = env or TrackingEnv()
        self.stateLow = self.env.stateLow[3:] + self.env.stateLow[:3]
        self.stateHigh = self.env.stateHigh[3:] + self.env.stateHigh[:3]
        self.actionLow = self.env.actionLow
        self.actionHigh = self.env.actionHigh
        self.actionDim = self.env.actionSpace.shape[0]
        self.stateDim = 6
        config = MPCConfig()
        self.gammar = config.gammar
        self.T = self.env.T
        state = ca.MX.sym("state", self.stateDim)
        action = ca.MX.sym("action", self.actionDim)
        self.T = 0.1
        self.m = 1520
        self.a = 1.19
        self.b = 1.46
        self.kf = -155_495
        self.kr = -155_495
        self.Iz = 2_642
        stateNextt = ca.vertcat(
            state[0] + self.T * (state[3] * ca.cos(state[2]) - state[4] * ca.sin(state[2])),
            state[1] + self.T * (state[4] * ca.cos(state[2]) + state[3] * ca.sin(state[2])),
            state[2] + self.T * state[5],
            state[3] + self.T * action[0],
            (
                (-(self.a * self.kf - self.b * self.kr) * state[5] + self.kf * action[1] * state[3])
                + self.m * state[5] * state[3] * state[3]
                - self.m * state[3] * state[4] / self.T
            )
            / (self.kf + self.kr - self.m * state[3] / self.T),
            (
                -self.Iz * state[5] * state[3] / self.T
                - (self.a * self.kf - self.b * self.kr) * state[4]
                + self.a * self.kf * action[1] * state[3]
            )
            / ((self.a**2 * self.kf + self.b**2 * self.kr) - self.Iz * state[3] / self.T),
        )
        self.F = ca.Function("F", [state, action], [stateNextt])

        refState = ca.MX.sym("refState", 3 * self.env.refNum)
        cost = (
            15 * (state[0] - refState[0]) ** 2
            + 15 * (state[1] - refState[1]) ** 2
            + 10 * (state[2] - refState[2]) ** 2
            + 2 * action[0] ** 2
            + 2 * action[1] ** 2
        )
        self.calCost = ca.Function("calCost", [state, refState, action], [cost])

        relState = ca.vertcat(
            state[3],
            state[4],
            state[5],
            (refState[0] - state[0]) * ca.cos(state[2]) + (refState[1] - state[1]) * ca.sin(state[2]),
            (refState[1] - state[1]) * ca.cos(state[2]) - (refState[0] - state[0]) * ca.sin(state[2]),
            ca.cos(refState[2] - state[2]),
            ca.sin(refState[2] - state[2]),
        )
        relState_row = ca.transpose(relState)

        relState_multi = [relState[:3]]
        for i in range(self.env.refNum):
            relState_i = [
                (refState[3 * i] - state[0]) * ca.cos(state[2])
                + (refState[3 * i + 1] - state[1]) * ca.sin(state[2]),
                (refState[3 * i + 1] - state[1]) * ca.cos(state[2])
                - (refState[3 * i] - state[0]) * ca.sin(state[2]),
                ca.cos(refState[3 * i + 2] - state[2]),
                ca.sin(refState[3 * i + 2] - state[2]),
            ]
            relState_multi.extend(relState_i)
        relState_multi = ca.transpose(ca.vertcat(*relState_multi))

        self.getrefState = ca.Function("getrefState", [state, refState], [relState_row])
        self.getrefState_multi = ca.Function("getrefState_multi", [state, refState], [relState_multi])
        self.value = value
        self.value_multi = value_multi
        self._l4c_cache = (Path(".cache") / "l4casadi").resolve()
        self._l4c_cache.mkdir(parents=True, exist_ok=True)
        self._one_step_model = None
        self._multi_step_model = None

    def MPCSolver(
        self,
        initState: Sequence[float],
        refState: Sequence[float],
        predictStep: int,
        isReal: bool = True,
        info: Optional[Sequence[float]] = None,
        terminalCost: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x: list = []
        lbx: list = []
        ubx: list = []
        lbg: list = []
        ubg: list = []
        G: list = []
        J = 0
        Xk = ca.MX.sym("X0", self.stateDim)
        x += [Xk]
        lbx += initState
        ubx += initState
        gammar = 1
        if info is not None:
            reft = info[0]
            refID = info[1]
        else:
            reft = 0.0
            refID = 0.0
        refState_ = list(refState)
        for k in range(1, predictStep + 1):
            Uk = ca.MX.sym(f"U{k-1}", self.actionDim)
            x += [Uk]
            lbx += self.actionLow
            ubx += self.actionHigh
            J += self.calCost(Xk, refState_, Uk) * gammar
            gammar *= self.gammar
            if isReal:
                refState_[:-3] = refState_[3:]
                refState_[-3:] = [
                    self.env.trajectoryList.calx(reft + self.env.refNum * self.env.T, refID, MPCflag=1),
                    self.env.trajectoryList.caly(reft + self.env.refNum * self.env.T, refID, MPCflag=1),
                    self.env.trajectoryList.calphi(reft + self.env.refNum * self.env.T, refID, MPCflag=1),
                ]
                reft += self.env.T
            else:
                refState_ = self.env.refDynamicVirtual(refState_, MPCflag=1)

            XNext = self.F(Xk, Uk)
            Xk = ca.MX.sym(f"X{k}", self.stateDim)
            G += [XNext - Xk]
            lbg += [0 for _ in range(self.stateDim)]
            ubg += [0 for _ in range(self.stateDim)]
            x += [Xk]
            lbx += self.stateLow
            ubx += self.stateHigh
        if terminalCost == "one-step":
            if self.value is None:
                raise ValueError("One-step terminal cost requested but no critic provided.")
            l4c_model = self._get_l4c_model("one-step")
            nn_input = self.getrefState(Xk, refState_)
            J += gammar * ca.sum1(l4c_model(nn_input))
        elif terminalCost == "multi-step":
            if self.value_multi is None:
                raise ValueError("Multi-step terminal cost requested but no critic provided.")
            l4c_model = self._get_l4c_model("multi-step")
            nn_input = self.getrefState_multi(Xk, refState_)
            J += gammar * ca.sum1(l4c_model(nn_input))

        nlp = dict(f=J, g=ca.vertcat(*G), x=ca.vertcat(*x))
        solver = ca.nlpsol("res", "ipopt", nlp, self._sol_dic)
        res = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=0)
        resX = np.array(res["x"])
        resState = np.zeros([predictStep, self.stateDim], dtype="float32")
        resControl = np.zeros([predictStep, self.actionDim], dtype="float32")
        totalDim = self.stateDim + self.actionDim
        for i in range(predictStep):
            resState[i] = resX[totalDim * i : totalDim * i + self.stateDim].reshape(-1)
            resControl[i] = resX[totalDim * i + self.stateDim : totalDim * (i + 1)].reshape(-1)
        return resState, resControl

    def _get_l4c_model(self, mode: str):
        if mode == "one-step":
            if self._one_step_model is None:
                build_dir = self._l4c_cache / "critic_one_step"
                build_dir.mkdir(parents=True, exist_ok=True)
                self._one_step_model = l4c.L4CasADi(
                    self.value,
                    device="cpu",
                    name="critic_one_step",
                    build_dir=str(build_dir),
                )
            return self._one_step_model
        if mode == "multi-step":
            if self._multi_step_model is None:
                build_dir = self._l4c_cache / f"critic_ref_{self.env.refNum}"
                build_dir.mkdir(parents=True, exist_ok=True)
                self._multi_step_model = l4c.L4CasADi(
                    self.value_multi,
                    device="cpu",
                    name=f"critic_multi_{self.env.refNum}",
                    build_dir=str(build_dir),
                )
            return self._multi_step_model
        raise ValueError(f"Unsupported terminal cost type: {mode}")
