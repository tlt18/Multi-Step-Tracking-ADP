from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

dir_list = {\
    1: [
        'Results_dir/2023-02-13-16-26-22/train/events.out.tfevents.1676276782.idlab-Server2.18816.0',
        'Results_dir/2023-02-14-10-40-57/train/events.out.tfevents.1676342457.idlab-Server2.553.0',
        'Results_dir/2023-02-15-10-19-07/train/events.out.tfevents.1676427547.idlab-Server2.14314.0',
        'Results_dir/2023-02-15-20-05-50/train/events.out.tfevents.1676462750.idlab-Server2.7416.0',
        'Results_dir/2023-02-16-09-09-52/train/events.out.tfevents.1676509792.idlab-Server2.16161.0'
        ],
    3: [
        'Results_dir/2023-02-13-16-29-00/train/events.out.tfevents.1676276940.idlab-Server2.19127.0',
        'Results_dir/2023-02-14-10-41-07/train/events.out.tfevents.1676342467.idlab-Server2.670.0',
        'Results_dir/2023-02-15-10-19-27/train/events.out.tfevents.1676427567.idlab-Server2.14435.0',
        'Results_dir/2023-02-15-20-05-58/train/events.out.tfevents.1676462758.idlab-Server2.7532.0',
        'Results_dir/2023-02-16-09-10-04/train/events.out.tfevents.1676509804.idlab-Server2.16278.0'
        ],
    5: [
        'Results_dir/2023-02-13-16-29-28/train/events.out.tfevents.1676276968.idlab-Server2.19344.0',
        'Results_dir/2023-02-14-10-41-17/train/events.out.tfevents.1676342477.idlab-Server2.789.0',
        'Results_dir/2023-02-15-10-19-35/train/events.out.tfevents.1676427575.idlab-Server2.14549.0',
        'Results_dir/2023-02-15-20-06-06/train/events.out.tfevents.1676462766.idlab-Server2.7685.0',
        'Results_dir/2023-02-16-09-10-14/train/events.out.tfevents.1676509814.idlab-Server2.16371.0'
        ],
    7: [
        'Results_dir/2023-02-13-16-29-47/train/events.out.tfevents.1676276987.idlab-Server2.19579.0',
        'Results_dir/2023-02-14-10-41-32/train/events.out.tfevents.1676342492.idlab-Server2.871.0',
        'Results_dir/2023-02-15-10-19-51/train/events.out.tfevents.1676427591.idlab-Server2.14678.0',
        'Results_dir/2023-02-15-20-06-15/train/events.out.tfevents.1676462775.idlab-Server2.7908.0',
        'Results_dir/2023-02-16-09-10-26/train/events.out.tfevents.1676509826.idlab-Server2.16477.0'
        ],
    9: [
        'Results_dir/2023-02-14-12-15-23/train/events.out.tfevents.1676348123.idlab-Server2.20782.0',
        'Results_dir/2023-02-14-10-41-41/train/events.out.tfevents.1676342501.idlab-Server2.1007.0',
        'Results_dir/2023-02-15-10-19-59/train/events.out.tfevents.1676427599.idlab-Server2.14780.0',
        'Results_dir/2023-02-15-20-06-22/train/events.out.tfevents.1676462782.idlab-Server2.8025.0',
        'Results_dir/2023-02-16-09-10-36/train/events.out.tfevents.1676509836.idlab-Server2.16593.0'
    ]
}

# refNum_list = [1]
refNum_list = [1, 3, 5, 7, 9]

for refNum in refNum_list:
    print('Plot case of refNum = {}'.format(refNum))
    dirs = dir_list[refNum]
    step_list = []
    value_list = []
    for dir in dirs:
        ea=event_accumulator.EventAccumulator(dir)
        ea.Reload()
        # print(ea.scalars.Keys())
        # ['Policy Loss', 'Value Loss', 'Virtual cost', 'Acc max error', 'Delta max error']
        data_item = ea.scalars.Items('Virtual cost')
        step_list += [i.step/1000 for i in data_item]
        temp_value = data_item[0].value 
        value_filter = []
        w = 0.0
        for i in data_item:
            temp_value = temp_value * w + i.value * (1-w)
            value_filter.append(temp_value)
        value_list += value_filter
    value_list = [v * 100 for v in value_list] # step
    data = pd.DataFrame.from_dict(
        {
            "step": step_list,
            "reward": value_list
        }
    )
    parameters = {
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        # 'figure.figsize': (9.0, 6.5),
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.unicode_minus': False}
    plt.rcParams.update(parameters)
    sns.set_style("darkgrid")
    plt.figure(0)
    sns.lineplot(data=data, x="step", y="reward", label = 'ADP(N='+str(refNum)+')')
    # plt.plot(data[:, 0]/1000, data[:, 1], label = 'ADP(N='+str(refNum)+')')
    plt.legend()
    plt.ylim(1.25, 4)
    # plt.yscale('log')
    plt.xlabel('Thousand Iteration')
    plt.ylabel('Cumulative cost')
    plt.savefig('./Results_dir/learning_curve/cumulative cost.png', bbox_inches='tight')

   