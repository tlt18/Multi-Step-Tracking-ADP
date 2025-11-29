"""Configuration objects used across training, simulation, and MPC."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

__all__ = ["MPCConfig", "trainConfig", "vehicleDynamicConfig"]


@dataclass
class trainConfig:
    """Hyper-parameters for the FAADP actor-critic training loop."""

    iterationMax: int = 30_000
    iterationPrint: int = 100
    iterationSave: int = 1_000
    lrPolicy: float = 1e-4
    lrValue: float = 1e-3
    stepForwardPEV: int = 30
    gammar: float = 0.95
    refNoise: float = 1.0
    lifeMax: int = 30
    batchSize: int = 256
    sampleSize: int = 256
    warmBuffer: int = 4 * 256
    capacity: int = 256_000
    tanLine: bool = False


@dataclass
class vehicleDynamicConfig:
    """Vehicle parameters and reference-trajectory definitions (SI units)."""

    refV: float = 5.0
    curveK: float = 1 / 6
    curveA: float = 1.0
    DLCh: float = 3.5
    DLCa: float = 30.0
    DLCb: float = 50.0
    curvePhi: float = np.pi / 60
    T: float = 0.1  # Sampling interval
    m: float = 1_520.0
    a: float = 1.19
    b: float = 1.46
    kf: float = -155_495.0
    kr: float = -155_495.0
    Iz: float = 2_642.0
    testStepReal: Dict[str, int] = field(
        default_factory=lambda: {
            "sine": 100,
            "DLC": 350,
            "TurnLeft": 30,
            "TurnRight": 30,
            "RandomTest": 500,
            "Circle": 400,
        }
    )
    testStepVirtual: int = 40
    testSampleNum: int = 1
    refNum: int = 9
    mpcstep: int = 60

    def __post_init__(self) -> None:
        self.initState: List[float] = [
            0.0,
            0.0,
            math.atan(self.curveA * self.curveK),
            self.refV,
            0.0,
            0.0,
        ]


@dataclass
class MPCConfig:
    """Problem definition for the MPC baselines."""

    MPCStep: List[int] = field(default_factory=lambda: [9])

    def __post_init__(self) -> None:
        self.gammar = trainConfig().gammar
