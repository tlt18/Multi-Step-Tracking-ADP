import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

parameters = {'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'axes.unicode_minus': False,
        'font.size': 18,
        'figure.figsize': (6, 4)
        }
plt.rcParams.update(parameters)

dir_list = [
    './Results_dir/refNum9/2023-03-06-09-53-05/simulationReal/sine',
    # 'Results_dir/refNum9/2023-03-06-09-53-05/simulationReal/DCL',
]
color_list = ['red', 'darkorange', 'limegreen', 'blue']
linestyle_list = ['-', '--', '-.', ':']  
marker_list = ['o', 's', 'D', '^']

item_dict = {
    'y-x': ['X [m]', 'Y [m]'], 
    'distance-error-t-cum': ['Time [s]', ' Cumu. distance error'],
    'phi-error-t-cum': ['Time [s]', ' Cumu. angle error'],
    'utility-t-cum': ['Time [s]', ' Cost'],
}

alg_dict = {
    "": "ADP",
    "-MPC-9_wo_TC": "MPC-9 w/o TC",
    "-MPC-9_w_1-step_TC": "MPC-9 w/ 1-step TC",
    "-MPC-9_w_9-step_TC": "MPC-9 w/ 9-step TC",
}

for dir in dir_list:
    for item_key, item_value in item_dict.items():
        fig, ax = plt.subplots(1, 1)
        # if item_key == 'y-x':
        #     file = f"{dir}/{item_key}-ref.csv"
        #     if not os.path.exists(file):
        #         print(f"File {file} not exists.")
        #     data = pd.read_csv(file)
        #     ax.plot(data.iloc[:, 0], data.iloc[:, 1], label='ref', color='black')
        for alg_key, alg_value in alg_dict.items():
            file = f"{dir}/{item_key}{alg_key}.csv"
            if not os.path.exists(file):
                print(f"File {file} not exists.")
            data = pd.read_csv(file, header=None, skiprows=1)
            x_value = data.iloc[0, :].values
            y_value = data.iloc[1, :].values
            color = color_list[list(alg_dict.keys()).index(alg_key)]
            linestyle = linestyle_list[list(alg_dict.keys()).index(alg_key)]
            marker = marker_list[list(alg_dict.keys()).index(alg_key) % len(marker_list)]
            ax.plot(x_value, y_value, label=alg_value, color=color)
        ax.set_xlabel(item_value[0])
        ax.set_ylabel(item_value[1])
        ax.legend(framealpha=0.5)
        plt.savefig(f"{dir}/{item_key}.pdf", bbox_inches='tight', dpi=300)
