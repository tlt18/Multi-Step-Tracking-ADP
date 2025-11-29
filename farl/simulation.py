"""Simulation utilities for comparing FAADP with MPC baselines."""

from __future__ import annotations

import argparse
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from .config import MPCConfig
from .env import TrackingEnv
from .networks import Actor, Critic
from .solver import Solver


def _calc_distance_error(state: np.ndarray) -> np.ndarray:
    """Return Euclidean position error between ego pose and reference pose."""
    return np.sqrt(
        np.power(state[:, 0] - state[:, 6], 2) + np.power(state[:, 1] - state[:, 7], 2)
    )


def _compute_relative_errors(
    state_adp: np.ndarray,
    control_adp: np.ndarray,
    reward_adp: np.ndarray,
    state_mpc: np.ndarray,
    control_mpc: np.ndarray,
    reward_mpc: np.ndarray,
    simu_dir: str,
) -> List[float]:
    """Return flattened [mean, max, ...] relative-error metrics."""
    metrics = [
        ("Acceleration", control_adp[:, 0], control_mpc[:, 0]),
        ("Steering Angle", control_adp[:, 1], control_mpc[:, 1]),
        ("Distance Error", _calc_distance_error(state_adp), _calc_distance_error(state_mpc)),
        ("Heading Angle Error", state_adp[:, 2] - state_adp[:, 8], state_mpc[:, 2] - state_mpc[:, 8]),
        ("Utility  Function", reward_adp, reward_mpc),
    ]
    error_values: List[float] = []
    for title, adp_data, mpc_data in metrics:
        mean_err, max_err = calRelError(adp_data, mpc_data, title=title, simu_dir=simu_dir)
        error_values.extend([mean_err, max_err])
    return error_values


REL_ERROR_HEADER = (
    "Acceleration mean,max,Steering Angle mean,max,Distance Error mean,max,"
    "Heading Angle mean,max,Utility Function mean, max"
)

def _save_rollout(path, env, states, controls):
    data = np.concatenate((states, controls), axis=1)
    header = "x,y,phi,u,v,omega," + "xr,yr,phir," * env.refNum + "a,delta"
    with open(path, 'wb') as f:
        np.savetxt(f, data, delimiter=',', fmt='%.4f', comments='', header=header)


def _initial_state(env: TrackingEnv, curve_type: str, seed: int):
    mapping = {
        "sine": 0,
        "DLC": 1,
        "Circle": 2,
        "RandomTest": 3,
    }
    if curve_type not in mapping:
        raise ValueError(f"Unsupported curve type: {curve_type}")
    return env.resetSpecific(1, noise=-1, refIDinit=mapping[curve_type], seed=seed, random_offset=False)


def _rollout_adp(env, policy, initial_state, info, steps, plot_delete):
    state = initial_state.clone()
    info_local = info.clone()
    state_records: List[np.ndarray] = []
    control_records: List[np.ndarray] = []
    reward_records: List[float] = []
    time_records: List[float] = []
    for _ in trange(steps, desc="Processing", unit="step"):
        current = state[0].detach().numpy()
        state_records.append(np.concatenate([current[-3:], current[:-3]]))
        rel_state = env.relStateCal(state)
        start = time.time()
        control = policy(rel_state).detach()
        time_records.append(time.time() - start)
        state, reward, _, info_local = env.stepSpecificRef(state, control, info_local)
        control_records.append(control[0].detach().numpy())
        reward_records.append(reward.item())
    state_arr = np.asarray(state_records)
    control_arr = np.asarray(control_records)
    reward_arr = np.asarray(reward_records)
    time_arr = np.asarray(time_records)
    if plot_delete:
        state_arr = state_arr[plot_delete:]
        control_arr = control_arr[plot_delete:]
        reward_arr = reward_arr[plot_delete:]
        time_arr = time_arr[plot_delete:]
    return state_arr, control_arr, reward_arr, time_arr


def _run_mpc_variant(env, solver, initial_state, info, mpcstep, curve_type, terminal_cost, plot_delete):
    env.randomTestReset()
    tempstate = initial_state[0].tolist()
    info_mpc = info[0].tolist()
    state_mpc = tempstate[-3:] + tempstate[:3]
    ref_state = tempstate[3:-3]
    state_records: List[np.ndarray] = []
    control_records: List[np.ndarray] = []
    reward_records: List[float] = []
    time_records: List[float] = []
    for _ in trange(env.testStepReal[curve_type], desc="Processing", unit="step"):
        start = time.time()
        _, control = solver.MPCSolver(state_mpc, ref_state, mpcstep, isReal=True, info=info_mpc, terminalCost=terminal_cost)
        time_records.append(time.time() - start)
        state_records.append(np.concatenate([np.asarray(state_mpc), np.asarray(ref_state)]))
        action = control[0].tolist()
        reward = env.calReward(state_mpc[-3:] + ref_state + state_mpc[:3], action, MPCflag=1)
        reward_records.append(float(reward))
        state_mpc = env.vehicleDynamic(
            state_mpc[0], state_mpc[1], state_mpc[2], state_mpc[3], state_mpc[4], state_mpc[5], action[0], action[1], MPCflag=1
        )
        ref_state[:-3] = ref_state[3:]
        ref_state[-3:] = [
            env.trajectoryList.calx(info_mpc[0] + env.refNum * env.T, info_mpc[1], MPCflag=1),
            env.trajectoryList.caly(info_mpc[0] + env.refNum * env.T, info_mpc[1], MPCflag=1),
            env.trajectoryList.calphi(info_mpc[0] + env.refNum * env.T, info_mpc[1], MPCflag=1),
        ]
        info_mpc[0] += env.T
        control_records.append(control[0])
    state_arr = np.asarray(state_records)
    control_arr = np.asarray(control_records)
    reward_arr = np.asarray(reward_records)
    time_arr = np.asarray(time_records)
    if plot_delete:
        state_arr = state_arr[plot_delete:]
        control_arr = control_arr[plot_delete:]
        reward_arr = reward_arr[plot_delete:]
        time_arr = time_arr[plot_delete:]
    return state_arr, control_arr, reward_arr, time_arr


def _nan_result(env, curve_type):
    steps = env.testStepReal[curve_type]
    action_dim = env.actionSpace.shape[0]
    return (
        np.full((steps, env.stateDim), np.nan),
        np.full((steps, action_dim), np.nan),
        np.full(steps, np.nan),
        np.full(steps, np.nan),
    )


def simulationReal(MPCStep, ADP_dir, simu_dir, refNum = None, curveType = 'sine', seed = 0, one_step_value_dir: Optional[str] = None):
    """Compare FAADP and several MPC variants on a specified reference curve."""
    print("----------------------Curve Type: {}----------------------".format(curveType))
    plotDelete = 0
    env = TrackingEnv()
    env.seed(seed)
    if refNum != None:
        env.changeRefNum(refNum)
    relstateDim = env.relstateDim
    actionDim = env.actionSpace.shape[0]
    policy = Actor(relstateDim, actionDim)
    policy.loadParameters(ADP_dir)

    value_one_step = None
    if one_step_value_dir:
        value_one_step = Critic(7, 1)
        value_one_step.loadParameters(one_step_value_dir)

    value_multi = Critic(relstateDim, 1)
    value_multi.loadParameters(ADP_dir)

    solver = Solver(env, value_one_step, value_multi)
    initialState, info = _initial_state(env, curveType, seed)

    stateADPList, controlADPList, rewardADP, timeADP = _rollout_adp(
        env, policy, initialState, info, env.testStepReal[curveType], plotDelete
    )
    saveADP = np.concatenate((stateADPList, controlADPList), axis = 1)
    with open(simu_dir + "/simulationRealADP.csv", 'wb') as f:
        np.savetxt(f, saveADP, delimiter=',', fmt='%.4f', comments='', header="x,y,phi,u,v,omega," + "xr,yr,phir,"*env.refNum + "a,delta")

    # MPC without terminal cost
    controlMPCAll: List[np.ndarray] = []
    stateMPCAll: List[np.ndarray] = []
    rewardMPCAll: List[np.ndarray] = []
    timeMPCAll: List[np.ndarray] = []
    labelMPCAll: List[str] = []

    def append_result(label, result):
        labelMPCAll.append(label)
        stateMPCAll.append(result[0])
        controlMPCAll.append(result[1])
        rewardMPCAll.append(result[2])
        timeMPCAll.append(result[3])

    for mpcstep in MPCStep:
        if mpcstep == 1:
            append_result(f"MPC-{mpcstep} w/o TC", _nan_result(env, curveType))
        else:
            result = _run_mpc_variant(env, solver, initialState, info, mpcstep, curveType, None, plotDelete)
            append_result(f"MPC-{mpcstep} w/o TC", result)
            _save_rollout(os.path.join(simu_dir, f"simulationRealMPC_{mpcstep}.csv"), env, result[0], result[1])
        if value_one_step is not None:
            result = _run_mpc_variant(
                env, solver, initialState, info, mpcstep, curveType, "one-step", plotDelete
            )
            append_result(f"MPC-{mpcstep} w/ 1-step TC", result)
            _save_rollout(os.path.join(simu_dir, f"simulationRealMPCTerminal_{mpcstep}.csv"), env, result[0], result[1])
        else:
            print(f"Skipping MPC-{mpcstep} w/ 1-step TC (no value provided)")
        result = _run_mpc_variant(
            env, solver, initialState, info, mpcstep, curveType, "multi-step", plotDelete
        )
        append_result(f"MPC-{mpcstep} w/ {mpcstep}-step TC", result)
        _save_rollout(
            os.path.join(simu_dir, f"simulationRealMPC_w_N-step_Terminal_{mpcstep}.csv"), env, result[0], result[1]
        )

    print("Time consume ADP: {}ms".format(timeADP.mean() * 1000))
    for label in labelMPCAll:
        print(f"Time consume {label}: {timeMPCAll[labelMPCAll.index(label)].mean() * 1000}ms")
    time_return = [timeADP.mean()] + [timempc.mean() for timempc in timeMPCAll]

    colorList = ['darkorange', 'green', 'blue', 'red']
    plt.figure()
    pos = list(range(len(labelMPCAll) + 1))
    labels = ["ADP"] + labelMPCAll
    time_values = [timeADP.mean() * 1000] + [timempc.mean() * 1000 for timempc in timeMPCAll]
    plt.bar(pos, time_values, width=0.3, color=[colorList[-1]] + [colorList[i] for i in range(len(labelMPCAll))], label=labels)
    plt.xticks(pos, labels)

    for x, y in enumerate(time_values):
        plt.text(x, y, '%s ms' % round(y, 2), ha='center', va='bottom', fontsize=9)

    plt.ylabel("Average calculation time [ms]")
    plt.yscale('log')
    plt.legend()
    plt.savefig(simu_dir + '/average-calculation-time.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    for i, label in enumerate(labelMPCAll):
        plt.plot(range(len(timeMPCAll[i])), timeMPCAll[i] * 1000, label = label, color = colorList[i])
    plt.plot(range(len(timeADP)), timeADP * 1000, label = 'ADP', color = colorList[-1])
    plt.legend()
    plt.ylabel("Calculation time [ms]")
    plt.xlabel('Step')
    plt.yscale('log')
    # plt.title("Calculation time")
    plt.savefig(simu_dir + '/calculation-time-step.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.boxplot([time * 1000 for time in timeMPCAll] + [timeADP * 1000],
                patch_artist=True,
                widths=0.4,
                showmeans=True,
                meanprops={'marker':'+',
                        'markerfacecolor':'k',
                        'markeredgecolor':'k',
                        'markersize':5})
    plt.xticks(range(1, len(labelMPCAll) + 2, 1),
            labelMPCAll + ['RL'])
    # plt.ylim(0,9)
    plt.yscale('log')
    plt.grid(axis='y',ls='--',alpha=0.5)
    plt.ylabel('Calculation time [ms]',fontsize=18)
    plt.savefig(simu_dir + '/boxplot-time.png', bbox_inches='tight')
    plt.close()

    # stateADPList: [x,y,phi,u,v,omega,[xr,yr,phir]]
    # controlMPCAll: [a, delta]
    # Cal relative error
    errorSaveList = _compute_relative_errors(
        stateADPList,
        controlADPList,
        rewardADP,
        stateMPCAll[-1],
        controlMPCAll[-1],
        rewardMPCAll[-1],
        simu_dir,
    )
    rel_error_file = Path(simu_dir) / "RelError.csv"
    with open(rel_error_file, "ab") as f:
        np.savetxt(
            f,
            np.array([errorSaveList]),
            delimiter=",",
            fmt="%.4f",
            comments="" if rel_error_file.exists() else "",
            header=REL_ERROR_HEADER if not rel_error_file.exists() else "",
        )

    # Plot
    # stateAll: [x,y,phi,u,v,omega,[xr,yr,phir]]
    # controlAll: [a, delta]
    figSize = (20,5)
    # y vs. x
    xADP = stateADPList[:,0]
    xMPC = [mpc[:,0] for mpc in stateMPCAll]
    xRef = stateADPList[:,6]
    yADP = stateADPList[:,1]
    yMPC = [mpc[:,1] for mpc in stateMPCAll]
    yRef = stateADPList[:,7]
    xName = 'X [m]'
    yName = 'Y [m]'
    title = 'y-x'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title, isRef = True, xRef = xRef, yRef = yRef, figSize='equal', lineWidth = 2)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title, isRef = True, xRef = xRef, yRef = yRef, lineWidth = 2)

    np.savetxt(simu_dir + f"/{title}.csv", np.array([xADP, yADP]), delimiter=',', fmt='%.4f', comments='', header=f"{xName},{yName}")
    np.savetxt(simu_dir + f"/{title}-ref.csv", np.array([xRef, yRef]), delimiter=',', fmt='%.4f', comments='', header=f"{xName},{yName}")
    for idx, label in enumerate(labelMPCAll):
        safe_label = label.replace(' ', '_').replace('/', '')
        np.savetxt(simu_dir + f"/{title}-{safe_label}.csv", np.array([xMPC[idx], yMPC[idx]]), delimiter=',', fmt='%.4f', comments='', header=f"{xName},{yName}")

    # distance error vs. t
    yADP = np.sqrt(np.power(stateADPList[:, 0] - stateADPList[:, 6], 2) + np.power(stateADPList[:, 1] - stateADPList[:, 7], 2))*100
    yMPC = [np.sqrt(np.power(mpc[:, 0] - mpc[:, 6], 2) + np.power(mpc[:, 1] - mpc[:, 7], 2))*100 for mpc in stateMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'Time [s]'
    yName = 'Distance error [cm]'
    title = 'distance-error-t'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title, figSize=figSize)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title)
    np.savetxt(simu_dir + f"/{title}-cum.csv", np.array([xADP, np.cumsum(yADP)]), delimiter=',', fmt='%.4f', comments='', header=f"{xName},{yName}")
    for idx, label in enumerate(labelMPCAll):
        safe_label = label.replace(' ', '_').replace('/', '')
        np.savetxt(simu_dir + f"/{title}-cum-{safe_label}.csv", np.array([xMPC[idx], np.cumsum(yMPC[idx])]), delimiter=',', fmt='%.4f', comments='', header=f"{xName},{yName}")

    Ip_ADP = np.sqrt(np.mean(np.power(stateADPList[:, 0] - stateADPList[:, 6], 2) + np.power(stateADPList[:, 1] - stateADPList[:, 7], 2)))
    Ip_MPC = [np.sqrt(np.mean(np.power(mpc[:, 0] - mpc[:, 6], 2) + np.power(mpc[:, 1] - mpc[:, 7], 2))) for mpc in stateMPCAll]

    print('Position error ADP: {}m'.format(Ip_ADP))
    for idx, label in enumerate(labelMPCAll):
        print('Position error {}: {}m'.format(label, Ip_MPC[idx]))

    # x error vs. t
    yADP = stateADPList[:, 0] - stateADPList[:, 6]
    yMPC = [mpc[:, 0] - mpc[:, 6] for mpc in stateMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'time [s]'
    yName = 'X error [m]'
    title = 'x-error-t'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title, figSize=figSize)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title)

    # y error vs. t
    yADP = stateADPList[:, 1] - stateADPList[:, 7]
    yMPC = [mpc[:, 1] - mpc[:, 7] for mpc in stateMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'time [s]'
    yName = 'Y error [m]'
    title = 'y-error-t'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title, figSize=figSize)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title)

    # phi vs. t
    yADP = stateADPList[:,2] * 180/np.pi
    yMPC = [mpc[:,2] * 180/np.pi for mpc in stateMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'time [s]'
    yName = 'Heading angle [deg]'
    title = 'phi-t'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title, figSize=figSize)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title)

    # phi error vs. t
    yADP = stateADPList[:,2] * 180/np.pi - stateADPList[:,8] * 180/np.pi
    yMPC = [mpc[:,2] * 180/np.pi - mpc[:,8] * 180/np.pi for mpc in stateMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'time [s]'
    yName = 'Heading angle error [deg]'
    title = 'phi-error-t'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title, figSize=figSize)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title)

    np.savetxt(simu_dir + f"/{title}-cum.csv", np.array([xADP, np.cumsum(np.abs(yADP))]), delimiter=',', fmt='%.4f', comments='', header=f"{xName},{yName}")
    for idx, label in enumerate(labelMPCAll):
        safe_label = label.replace(' ', '_').replace('/', '')
        np.savetxt(simu_dir + f"/{title}-cum-{safe_label}.csv", np.array([xMPC[idx], np.cumsum(np.abs(yMPC[idx]))]), delimiter=',', fmt='%.4f', comments='', header=f"{xName},{yName}")

    Iphi_ADP = np.sqrt(np.mean(np.power(stateADPList[:,2] * 180/np.pi - stateADPList[:,8] * 180/np.pi, 2)))
    Iphi_MPC = [np.sqrt(np.mean(np.power(mpc[:,2] * 180/np.pi - mpc[:,8] * 180/np.pi, 2))) for mpc in stateMPCAll]

    print('Phi error ADP: {} deg'.format(Iphi_ADP))
    for idx, label in enumerate(labelMPCAll):
        print('Phi error {}: {} deg'.format(label, Iphi_MPC[idx]))

    # utility vs. t
    yADP = rewardADP
    yMPC = [mpc for mpc in rewardMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'time [s]'
    yName = 'Utility'
    title = 'utility-t'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title, figSize=figSize)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title)

    np.savetxt(simu_dir + f"/{title}-cum.csv", np.array([xADP, np.cumsum(yADP)]), delimiter=',', fmt='%.4f', comments='', header=f"{xName},{yName}")
    for idx, label in enumerate(labelMPCAll):
        safe_label = label.replace(' ', '_').replace('/', '')
        np.savetxt(simu_dir + f"/{title}-cum-{safe_label}.csv", np.array([xMPC[idx], np.cumsum(yMPC[idx])]), delimiter=',', fmt='%.4f', comments='', header=f"{xName},{yName}")

    # accumulated utility vs. t
    yADP = np.cumsum(rewardADP)
    yMPC = [np.cumsum(mpc) for mpc in rewardMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'time [s]'
    yName = 'Accumulated utility'
    title = 'accumulated-utility-t'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title, figSize=figSize)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title)
    print('Accumulated utility of ADP {:.4f}, {:.4f}% higher than MPC'.format(yADP[-1], (yADP[-1]-yMPC[-1][-1])/yMPC[-1][-1]*100))
    for idx, label in enumerate(labelMPCAll):
        print('Accumulated utility of {}: {:.4f}, {:.4f}% higher than MPC'.format(label, yMPC[idx][-1], (yMPC[idx][-1]-yMPC[-1][-1])/yMPC[-1][-1]*100))
    tar_return = [yADP[-1]] + [mpc[-1] for mpc in yMPC]

    # a vs. t
    yADP = controlADPList[:,0]
    yMPC = [mpc[:,0] for mpc in controlMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'time [s]'
    yName = 'a [m/s^2]'
    title = 'a-t'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title, figSize=figSize)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title)

    # delta vs. t
    yADP = controlADPList[:,1] * 180/np.pi
    yMPC = [mpc[:,1] * 180/np.pi for mpc in controlMPCAll]
    xADP = np.arange(0, len(yADP)) * env.T
    xMPC = [np.arange(0, len(mpc)) * env.T for mpc in yMPC]
    xName = 'time [s]'
    yName = 'delta [deg]'
    title = 'delta-t'
    if curveType == 'RandomTest':
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title, figSize=figSize)
    else:
        comparePlot(xADP, xMPC, yADP, yMPC, labelMPCAll, xName, yName, simu_dir, title)

    return time_return, tar_return

def comparePlot(
    xADP,
    xMPC,
    yADP,
    yMPC,
    labelMPCAll,
    xName,
    yName,
    simu_dir,
    title,
    isMark = False,
    isError = False,
    isRef = False,
    xRef = None,
    yRef = None,
    figSize = None,
    lineWidth = 2,
):
    """Plot ADP vs MPC time-series or phase-space curves."""
    if figSize != None and figSize != 'equal':
        plt.figure(figsize=figSize, dpi=300)
    else:
        plt.figure()
    colorList = ['darkorange', 'limegreen', 'blue', 'red']
    if isMark == True:
        markerList = ['|', 'D', 'o', '*']
    else:
        markerList = ['None', 'None', 'None', 'None']
    for idx, labelMPC in enumerate(labelMPCAll):
        plt.plot(xMPC[idx], yMPC[idx], linewidth=lineWidth, color=colorList[idx], linestyle='--', marker=markerList[idx], markersize=4, label=labelMPC)

    plt.plot(xADP, yADP, linewidth = lineWidth, color=colorList[-1],linestyle = '--', marker=markerList[-1], markersize=4)

    if isError == True:
        plt.plot([np.min(xADP), np.max(xADP)], [0,0], linewidth = lineWidth/2, color = 'grey', linestyle = '--')
        plt.legend(labels=labelMPCAll + ['ADP', 'Ref'])
    elif isRef == True:
        plt.plot(xRef, yRef, linewidth = lineWidth/2, color = 'gray', linestyle = '--')
        plt.legend(labels=labelMPCAll + ['ADP', 'Ref'])
    else:
        plt.legend(labels=labelMPCAll + ['ADP'])
    plt.xlabel(xName)
    plt.ylabel(yName)
    # plt.savefig(simu_dir + '/' + title + '.png', bbox_inches='tight')
    plt.savefig(simu_dir + '/' + title + '.png')
    if figSize == 'equal':
        plt.axis('equal')
    else:
        plt.axis('scaled')
    plt.close()

def calRelError(ADP, MPC, title, simu_dir, isPlot = False, isPrint = True):
    """Compute mean/max relative error between FAADP and MPC traces."""
    maxMPC = np.max(MPC, 0)
    minMPC = np.min(MPC, 0)
    relativeError = np.abs((ADP - MPC)/(maxMPC - minMPC + 1e-3))
    relativeErrorMax = np.max(relativeError, 0)
    relativeErrorMean = np.mean(relativeError, 0)
    if isPrint == True:
        print(title +' Error | Mean: {:.4f}%, Max: {:.4f}%'.format(relativeErrorMean*100,relativeErrorMax*100))
    if isPlot == True:
        plt.figure()
        data = relativeError
        plt.hist(data, bins=30, weights = np.zeros_like(data) + 1 / len(data))
        plt.xlabel('Relative Error of '+title)
        plt.ylabel('Frequency')
        plt.title('Relative Error of '+title)
        plt.savefig(simu_dir + '/relative-error-'+title+'.png')
        plt.close()
    return relativeErrorMean, relativeErrorMax

def simuVirtualTraning(env, ADP_dir, noise = -1, refIDinit = 0):
    """Evaluate a trained policy in the simulator and return average reward."""
    config = MPCConfig()
    mpcstep = max(config.MPCStep)
    relstateDim = env.relstateDim
    actionDim = env.actionSpace.shape[0]
    # ADP
    policy = Actor(relstateDim, actionDim)
    policy.loadParameters(ADP_dir)
    count = 0
    stateAdp, infoADP = env.resetSpecific(env.testSampleNum, noise = noise, refIDinit = refIDinit)
    controlADPList = np.empty(0)
    rewardList = np.empty(0)
    if refIDinit == 0:
        testStep = env.testStepReal["sine"]
    elif refIDinit == 1:
        testStep = env.testStepReal["DLC"]
    while(count < testStep):
        relState = env.relStateCal(stateAdp)
        controlAdp = policy(relState).detach()
        stateAdp, reward, done, infoADP = env.stepSpecificRef(stateAdp, controlAdp, infoADP)
        controlADPList = np.append(controlADPList, controlAdp.numpy())
        rewardList = np.append(rewardList, reward.numpy().mean())
        count += 1
    controlADPList =np.reshape(controlADPList, (testStep, env.testSampleNum, actionDim))
    ADPAction = np.array(np.transpose(controlADPList, (1, 0, 2)))

    return rewardList.mean()

def main(ADP_dir, refNum, seed = 0, one_step_value_dir: Optional[str] = None):
    """Run sine/DLC/RandomTest simulations for a given checkpoint."""
    config = MPCConfig()
    MPCStep = [refNum]

    parameters = {'axes.labelsize': 20,
        'axes.titlesize': 18,
    #   'figure.figsize': (9.0, 6.5),
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'axes.unicode_minus': False,
        'font.size': 12.5,
        'figure.figsize': (10, 6.4)
        }
    plt.rcParams.update(parameters)

    return_dict = {
        "time_sine": [],
        "time_DLC": [],
        "time_randomTest": [],
        "tar_sine": [],
        "tar_DLC": [],
        "tar_randomTest": [],
    }

    simu_dir = ADP_dir + '/simulationReal/sine'
    os.makedirs(simu_dir, exist_ok=True)
    return_dict["time_sine"], return_dict["tar_sine"] = simulationReal(MPCStep, ADP_dir, simu_dir, refNum = refNum, curveType = 'sine', seed = seed, one_step_value_dir = one_step_value_dir)

    simu_dir = ADP_dir + '/simulationReal/DLC'
    os.makedirs(simu_dir, exist_ok=True)
    return_dict["time_DLC"], return_dict["tar_DLC"] = simulationReal(MPCStep, ADP_dir, simu_dir, refNum = refNum, curveType = 'DLC', seed = seed, one_step_value_dir = one_step_value_dir)

    simu_dir = ADP_dir + '/simulationReal/randomTest'
    os.makedirs(simu_dir, exist_ok=True)
    return_dict["time_randomTest"], return_dict["tar_randomTest"] = simulationReal(MPCStep, ADP_dir, simu_dir, refNum = refNum, curveType = 'RandomTest', seed = seed, one_step_value_dir = one_step_value_dir)

    return return_dict

def cli_main() -> None:
    """Simple CLI wrapper around :func:`main` for quick experiments."""
    parser = argparse.ArgumentParser(description="Run FAADP tracking simulations.")
    parser.add_argument('--adp_dir', type=str, required=True, help='Directory containing a trained FAADP checkpoint.')
    parser.add_argument('--num_experiments', type=int, default=1, help='How many different seeds to evaluate.')
    parser.add_argument('--one_step_value_dir', type=str, default=None, help='Optional critic checkpoint for 1-step terminal cost.')
    args = parser.parse_args()

    match = re.search(r'refNum(\d+)', args.adp_dir)
    if not match:
        raise ValueError(f"Cannot extract refNum from the directory path: {args.adp_dir}")
    refNum = int(match.group(1))

    for seed in range(args.num_experiments):
        print(f"\n=== Experiment {seed + 1}/{args.num_experiments} (refNum={refNum}) ===")
        main(args.adp_dir, refNum, seed=seed, one_step_value_dir=args.one_step_value_dir)


if __name__ == "__main__":
    cli_main()
