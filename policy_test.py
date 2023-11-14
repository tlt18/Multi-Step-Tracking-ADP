from myenv import TrackingEnv
from network import ActorForIDC
import torch.onnx
import pandas as pd

ADP_dir = './Results_dir/refNum21/2023-11-13-19-45-31'
nn_input_dir = '2023-11-14_10-26-57.csv'
env = TrackingEnv()
relstateDim = env.relstateDim
actionDim = env.actionSpace.shape[0]
policy = ActorForIDC(relstateDim, actionDim)
policy.loadParameters(ADP_dir + "/network")

raw_input = pd.read_csv('./nn_input/nn_input_tlt' + nn_input_dir, header=None)
# convert to torch.Tensor
stateForIDC = torch.tensor(raw_input.values)

nn_obs = policy.preprocess(stateForIDC)
nn_action = policy(stateForIDC)

store_data = torch.concat([nn_action, nn_obs], dim = 1).detach().numpy()
store_data = pd.DataFrame(store_data)

# [u, v, w, [delta_x, delta_y, cos delta_phi, sin delta_phi] * N]
header = ["delta", "acc", "u", "v", "w"]
for i in range(policy.refNum):
    header.append("delta_x_" + str(i))
    header.append("delta_y_" + str(i))
    header.append("cos_delta_phi_" + str(i))
    header.append("sin_delta_phi_" + str(i))
store_data.to_csv('./nn_output/nn_output_tlt' + nn_input_dir, header=header, index=False)
