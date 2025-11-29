from __future__ import annotations

from typing import Optional

import torch

from .config import trainConfig
from .replay_buffer import ReplayBuffer

__all__ = ["Train"]


class Train:
    """Implements the policy evaluation/improvement loop from the paper."""

    def __init__(self, env, log_dir: str):
        self.env = env
        config = trainConfig()
        self.stepForwardPEV = config.stepForwardPEV
        self.batchSize = config.batchSize
        self.sampleSize = config.sampleSize
        self.warmBuffer = config.warmBuffer
        self.gammar = config.gammar
        self.lifeMax = config.lifeMax
        self.refNoise = config.refNoise
        self.tanLine = config.tanLine
        self.statelifeMax = torch.rand(self.sampleSize) * config.lifeMax
        self.sampleData: Optional[torch.Tensor] = None
        self.sampleInfo: Optional[torch.Tensor] = None
        self.sampleDataLife = torch.zeros(self.sampleSize)
        self.accumulateReward = torch.zeros(self.batchSize)
        self.stateForwardNext: Optional[torch.Tensor] = None
        self.doneForward: Optional[torch.Tensor] = None
        self.gammarForward = torch.tensor(1.0)
        self.buffer = ReplayBuffer(config.capacity)
        self.lossIteraValue = torch.tensor(0.0)
        self.lossIteraPolicy = torch.tensor(0.0)
        self.reset()

    def reset(self) -> None:
        """Initialise the replay buffer with fresh samples."""
        self.sampleData, self.sampleInfo = self.env.resetSpecific(self.batchSize)
        stacked = [
            torch.cat([self.sampleData[i], self.sampleInfo[i]], dim=-1)
            for i in range(self.sampleSize)
        ]
        self.buffer.extend(stacked)

    def update(self, policy) -> None:
        """Roll the environment forward to refresh the replay buffer."""
        relState = self.env.relStateCal(self.sampleData)
        control = policy(relState).detach()
        self.sampleData, _, done, self.sampleInfo = self.env.stepSpecificRef(
            self.sampleData, control, self.sampleInfo
        )
        self.sampleDataLife += 1
        if done.any():
            mask = done == 1
            self.sampleData[mask], self.sampleInfo[mask] = self.env.resetSpecific(int(mask.sum()))
            self.sampleDataLife[mask] = 0
        expired = self.sampleDataLife > self.statelifeMax
        if expired.any():
            count = int(expired.sum())
            self.sampleData[expired], self.sampleInfo[expired] = self.env.resetSpecific(count)
            self.sampleDataLife[expired] = 0
            self.statelifeMax[expired] = torch.rand(count) * self.lifeMax
        stacked = [
            torch.cat([self.sampleData[i], self.sampleInfo[i]], dim=-1)
            for i in range(self.sampleSize)
        ]
        self.buffer.extend(stacked)

    def policyEvaluate(self, policy, value) -> None:
        """Estimate the critic target by forward rolling cached states."""
        while len(self.buffer) < self.warmBuffer:
            self.update(policy)

        batchData_ = torch.stack(self.buffer.sample(self.batchSize))
        self.batchData = batchData_[:, :-2]
        self.batchInfo = batchData_[:, -2:]

        relState = self.env.relStateCal(self.batchData)
        valuePredict = value(relState).view(self.batchSize)
        valueTarget = torch.zeros(self.batchSize, device=valuePredict.device)
        stateNext = self.batchData.clone()
        infoNext = self.batchInfo.clone()
        self.gammarForward = torch.tensor(1.0)

        for _ in range(self.stepForwardPEV):
            relState = self.env.relStateCal(stateNext)
            control = policy(relState)
            stateNext, reward, done, infoNext = self.env.stepSpecificRef(
                stateNext, control, infoNext, tanLine=self.tanLine
            )
        mask = (~done).float()
        valueTarget += reward * self.gammarForward * mask
        self.gammarForward = self.gammarForward * self.gammar

        self.accumulateReward = valueTarget.clone()
        self.stateForwardNext = stateNext.clone()
        self.doneForward = done
        relState = self.env.relStateCal(stateNext)
        next_value = value(relState).view(self.batchSize)
        valueTarget += (~done).float() * next_value * self.gammarForward
        lossValue = torch.pow(valuePredict - valueTarget.detach(), 2).mean()

        value.zero_grad()
        lossValue.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
        value.opt.step()
        value.scheduler.step()
        self.lossIteraValue = lossValue.detach().cpu()

    def policyImprove(self, policy, value) -> None:
        """Perform the policy-improvement step while keeping the critic frozen."""
        for param in value.parameters():
            param.requires_grad = False

        relState = self.env.relStateCal(self.stateForwardNext)
        critic_out = value(relState).view(self.accumulateReward.shape[0])
        valueTarget = self.accumulateReward + (~self.doneForward).float() * critic_out * self.gammarForward.detach()

        for param in value.parameters():
            param.requires_grad = True

        policy.zero_grad()
        lossPolicy = valueTarget.mean()
        lossPolicy.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        policy.opt.step()
        policy.scheduler.step()
        self.lossIteraPolicy = lossPolicy.detach().cpu()
