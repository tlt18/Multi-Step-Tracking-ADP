from sys import path
path.append(r"F:/casadi-windows-py38-v3.5.5-64bit")
import config
import matplotlib.pyplot as plt
import myenv

def simulation(MPCStep, simu_dir)

if __name__ == '__main__':
    config = MPCConfig()
    MPCStep = config.MPCStep
    simu_dir = "./Simulation_dir/" + datetime.now().strftime("%Y-%m-%d-%H-%M")
    os.makedirs(simu_dir, exist_ok=True)
    simulation(MPCStep, simu_dir)



