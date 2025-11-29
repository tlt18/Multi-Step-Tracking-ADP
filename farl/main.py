"""Entry points for training and evaluating FAADP agents."""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from .config import trainConfig
from .env import TrackingEnv
from .networks import Actor, Critic
from .training import Train
from . import simulation as simulation_module


def run(is_train: bool = True, is_simu: bool = True, seed: int = 60) -> None:
    """Train FAADP (and optionally run simulations) using default settings."""

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    torch.set_num_threads(4)

    config = trainConfig()
    env = TrackingEnv()
    env.seed(seed)

    refNum = env.refNum
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = Path("Results_dir") / f"refNum{refNum}" / timestamp
    train_dir = log_dir / "train"
    code_dir = log_dir / "code"
    train_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    shutil.copytree(repo_root / "farl", code_dir / "farl", dirs_exist_ok=True)

    relstateDim = env.relstateDim
    actionDim = env.actionSpace.shape[0]
    policy = Actor(relstateDim, actionDim, lr=config.lrPolicy)
    value = Critic(relstateDim, 1, lr=config.lrValue)

    data_writer = SummaryWriter(train_dir.as_posix())

    if is_train:
        print("----------------------Start Training!----------------------")
        trainer = Train(env, train_dir.as_posix())
        time_begin = time.time()
        for iteration in trange(config.iterationMax):
            trainer.policyEvaluate(policy, value)
            trainer.policyImprove(policy, value)
            trainer.update(policy)
            data_writer.add_scalar('Policy Loss', trainer.lossIteraPolicy.mean(), iteration)
            data_writer.add_scalar('Value Loss', trainer.lossIteraValue.mean(), iteration)
            if iteration % config.iterationSave == 0 or iteration == config.iterationMax - 1:
                print(
                    "iteration: {}, LossValue: {:.4f}, LossPolicy: {:.4f}, value lr: {:10f}, policy lr: {:10f}".format(
                        iteration,
                        trainer.lossIteraValue,
                        trainer.lossIteraPolicy,
                        value.opt.param_groups[0]['lr'],
                        policy.opt.param_groups[0]['lr'],
                    )
                )
                value.saveParameters(log_dir.as_posix())
                policy.saveParameters(log_dir.as_posix())
                for curve in ['sine', 'DLC', 'TurnLeft', 'TurnRight', 'RandomTest']:
                    env.policyTestReal(policy, iteration, train_dir.as_posix(), curveType=curve)
                reward_sine = simulation_module.simuVirtualTraning(env, log_dir.as_posix(), noise=-1, refIDinit=0)
                reward_dlc = simulation_module.simuVirtualTraning(env, log_dir.as_posix(), noise=-1, refIDinit=1)
                data_writer.add_scalar('Sine cost', reward_sine, iteration)
                data_writer.add_scalar('DLC cost', reward_dlc, iteration)
                print(f"Accumulated Cost in sine is {reward_sine:.4f}")
                print(f"Accumulated Cost in DLC is {reward_dlc:.4f}")
                time_delta = time.time() - time_begin
                h = time_delta // 3600
                mi = (time_delta - h * 3600) // 60
                sec = time_delta % 60
                print(f"Time consuming: {h:.0f}h {mi:.0f}min {sec:.0f}sec")

    if is_simu:
        simulation_module.main(log_dir.as_posix(), refNum)


def main() -> None:
    """CLI entry point."""
    run()


if __name__ == "__main__":
    main()
