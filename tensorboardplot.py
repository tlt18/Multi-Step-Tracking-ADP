from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

dir_list = {\
    1: [
        'Results_dir/refNum1/2023-03-06-09-52-34/train/events.out.tfevents.1678067554.idlab-Server2.24022.0',
        'Results_dir/refNum1/2023-03-07-15-21-50/train/events.out.tfevents.1678173710.idlab-Server2.32234.0',
        'Results_dir/refNum1/2023-03-08-14-12-40/train/events.out.tfevents.1678255960.idlab-Server2.24758.0',
        'Results_dir/refNum1/2023-03-13-11-28-07/train/events.out.tfevents.1678678087.idlab-Server2.773.0',
        'Results_dir/refNum1/2023-03-15-16-24-56/train/events.out.tfevents.1678868696.idlab-Server2.30290.0'
        ],
    3: [
        'Results_dir/refNum3/2023-03-07-15-22-02/train/events.out.tfevents.1678173722.idlab-Server2.32336.0',
        'Results_dir/refNum3/2023-03-07-15-22-41/train/events.out.tfevents.1678173761.idlab-Server2.32717.0',
        'Results_dir/refNum3/2023-03-08-14-12-48/train/events.out.tfevents.1678255968.idlab-Server2.24855.0',
        'Results_dir/refNum3/2023-03-10-09-52-50/train/events.out.tfevents.1678413170.idlab-Server2.12031.0',
        'Results_dir/refNum3/2023-03-11-13-43-42/train/events.out.tfevents.1678513422.idlab-Server2.19338.0'
        ],
    5: [
        'Results_dir/refNum5/2023-03-06-09-52-53/train/events.out.tfevents.1678067573.idlab-Server2.24231.0',
        'Results_dir/refNum5/2023-03-07-15-22-10/train/events.out.tfevents.1678173730.idlab-Server2.32429.0',
        'Results_dir/refNum5/2023-03-08-14-12-53/train/events.out.tfevents.1678255973.idlab-Server2.24948.0',
        'Results_dir/refNum5/2023-03-10-09-53-06/train/events.out.tfevents.1678413186.idlab-Server2.12510.0',
        'Results_dir/refNum5/2023-03-11-13-43-48/train/events.out.tfevents.1678513428.idlab-Server2.19423.0'
        ],
    7: [
        'Results_dir/refNum7/2023-03-06-09-53-00/train/events.out.tfevents.1678067580.idlab-Server2.24328.0',
        'Results_dir/refNum7/2023-03-07-15-22-18/train/events.out.tfevents.1678173738.idlab-Server2.32525.0',
        'Results_dir/refNum7/2023-03-08-14-12-59/train/events.out.tfevents.1678255979.idlab-Server2.25047.0',
        'Results_dir/refNum7/2023-03-10-09-53-13/train/events.out.tfevents.1678413193.idlab-Server2.12605.0',
        'Results_dir/refNum7/2023-03-11-13-43-52/train/events.out.tfevents.1678513432.idlab-Server2.19519.0'
        ],
    9: [
        'Results_dir/refNum9/2023-03-06-09-53-05/train/events.out.tfevents.1678067585.idlab-Server2.24428.0',
        'Results_dir/refNum9/2023-03-07-15-22-27/train/events.out.tfevents.1678173747.idlab-Server2.32618.0',
        'Results_dir/refNum9/2023-03-08-14-13-05/train/events.out.tfevents.1678255985.idlab-Server2.25143.0',
        'Results_dir/refNum9/2023-03-10-09-53-21/train/events.out.tfevents.1678413201.idlab-Server2.12688.0',
        'Results_dir/refNum9/2023-03-11-13-43-59/train/events.out.tfevents.1678513439.idlab-Server2.19618.0'
    ]
}


# refNum_list = [1]
refNum_list = [1, 3, 5, 7, 9]
max_iteration = 30001

def myPolt(type = 'DLC cost', ylimit = [0.1, 4]):
    for refNum in refNum_list:
        print('Plot case of refNum = {}'.format(refNum))
        dirs = dir_list[refNum]
        step_list = []
        value_list = []
        for dir in dirs:
            ea=event_accumulator.EventAccumulator(dir)
            ea.Reload()
            # print(ea.scalars.Keys())
            data_item = ea.scalars.Items(type)
            step_add = [i.step/1000 for i in data_item if i.step < max_iteration]
            step_list += step_add
            temp_value = data_item[0].value 
            value_filter = []
            w = 0.0
            for i in data_item:
                temp_value = temp_value * w + i.value * (1-w)
                value_filter.append(temp_value)
            value_filter = value_filter[:len(step_add)]
            value_list += value_filter
        value_list = [v * 100 for v in value_list] # step

        data = pd.DataFrame.from_dict(
            {
                "step": step_list,
                "reward": value_list
            }
        )
        sns.set_style("darkgrid")
        sns.lineplot(data=data, x="step", y="reward", label = 'ADP(N='+str(refNum)+')')
        # plt.plot(data[:, 0]/1000, data[:, 1], label = 'ADP(N='+str(refNum)+')')
    plt.legend()
    # plt.ylim(1.25, 4)
    plt.ylim(ylimit)
    # plt.yscale('log')
    plt.xlabel('Thousand Iteration')
    plt.ylabel(type)
    plt.savefig('./Results_dir/learning_curve/'+type+'.png', bbox_inches='tight')


if __name__ == '__main__':
    parameters = {
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'figure.figsize': (6, 4),
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.unicode_minus': False}
    plt.rcParams.update(parameters)

    plt.figure(0)
    myPolt('DLC cost', ylimit = [0.1, 4])
    plt.figure(1)
    myPolt('Sine cost', ylimit = [1.25, 4])