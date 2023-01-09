from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np

dir_list = [\
    'Results_dir/2022-11-16-15-41-50/train/events.out.tfevents.1668584510.idlab-Server2.2178.0',\
    'Results_dir/2022-10-10-10-37-05/train/events.out.tfevents.1665369425.idlab-Server2.3001.0',\
    'Results_dir/2022-10-10-19-49-19/train/events.out.tfevents.1665402559.idlab-Server2.13023.0',\
    'Results_dir/2022-10-13-12-03-03/train/events.out.tfevents.1665633783.idlab-Server2.17626.0',\
    'Results_dir/2022-10-13-00-46-36/train/events.out.tfevents.1665593196.idlab-Server2.23746.0',\
    'Results_dir/2022-10-13-12-03-24/train/events.out.tfevents.1665633804.idlab-Server2.17967.0',\
    'Results_dir/2022-10-14-23-12-25/train/events.out.tfevents.1665760345.idlab-Server2.22880.0',\
    'Results_dir/2022-10-15-16-56-55/train/events.out.tfevents.1665824215.idlab-Server2.19369.0',\
    'Results_dir/2022-11-18-11-06-35/train/events.out.tfevents.1668740795.idlab-Server2.27952.0'
    ]
refNum_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]

for dir, refNum in zip(dir_list, refNum_list):
    print('Plot case of refNum = {}'.format(refNum))
    ea=event_accumulator.EventAccumulator(dir)
    ea.Reload()
    # print(ea.scalars.Keys())
    # ['Policy Loss', 'Value Loss', 'Virtual cost', 'Acc max error', 'Delta max error']

    # Policy Loss
    plt.figure(1)
    policy_loss_item = ea.scalars.Items('Policy Loss')
    policy_loss = np.array([[i.step, i.value] for i in policy_loss_item])
    policy_loss = np.reshape(policy_loss, (-1, 2))
    plt.plot(policy_loss[:, 0]/1000, policy_loss[:, 1], label = 'RefNum='+str(refNum))
    plt.legend()
    plt.ylim(25, 100)
    plt.yscale('log')
    plt.xlabel('Thousand Iteration')
    plt.ylabel('Policy Loss')
    plt.savefig('./Results_dir/learning_curve/policy_loss.png')

    # Value Loss
    plt.figure(2)
    policy_loss_item = ea.scalars.Items('Value Loss')
    policy_loss = np.array([[i.step, i.value] for i in policy_loss_item])
    policy_loss = np.reshape(policy_loss, (-1, 2))
    plt.plot(policy_loss[:, 0]/1000, policy_loss[:, 1], label = 'RefNum='+str(refNum))
    plt.legend()
    plt.ylim(0.1, 100)
    plt.yscale('log')
    plt.xlabel('Thousand Iteration')
    plt.ylabel('Value Loss')
    plt.savefig('./Results_dir/learning_curve/value_loss.png')

    # Virtual cost
    plt.figure(3)
    policy_loss_item = ea.scalars.Items('Virtual cost')
    policy_loss = np.array([[i.step, i.value] for i in policy_loss_item])
    policy_loss = np.reshape(policy_loss, (-1, 2))
    plt.plot(policy_loss[:, 0]/1000, policy_loss[:, 1], label = 'RefNum='+str(refNum))
    plt.legend()
    plt.ylim(1.3, 4)
    plt.yscale('log')
    plt.xlabel('Thousand Iteration')
    plt.ylabel('Virtual cost')
    plt.savefig('./Results_dir/learning_curve/virtual_cost.png')

    # Acc max error
    plt.figure(4)
    policy_loss_item = ea.scalars.Items('Acc max error')
    policy_loss = np.array([[i.step, i.value] for i in policy_loss_item])
    policy_loss = np.reshape(policy_loss, (-1, 2))
    plt.plot(policy_loss[:, 0]/1000, policy_loss[:, 1] * 100, label = 'RefNum='+str(refNum))
    plt.legend()
    plt.ylim(0.8, 20)
    plt.xlabel('Thousand Iteration')
    plt.ylabel('Maximal relative acceleration error [%]')
    plt.savefig('./Results_dir/learning_curve/acc_max_error.png')

    # Delta max error
    plt.figure(5)
    policy_loss_item = ea.scalars.Items('Delta max error')
    policy_loss = np.array([[i.step, i.value] for i in policy_loss_item])
    policy_loss = np.reshape(policy_loss, (-1, 2))
    plt.plot(policy_loss[:, 0]/1000, policy_loss[:, 1] * 100, label = 'RefNum='+str(refNum))
    plt.legend()
    plt.ylim(0, 20)
    plt.xlabel('Thousand Iteration')
    plt.ylabel('Maximal relative steering angle error [%]')
    plt.savefig('./Results_dir/learning_curve/delta_max_error.png')
