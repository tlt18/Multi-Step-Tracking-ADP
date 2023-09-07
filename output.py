from myenv import TrackingEnv
from network import Actor, Critic
import torch.onnx
import onnx

ADP_dir = './Results_dir/2022-05-26-15-15-14'
env = TrackingEnv()
relstateDim = env.relstateDim
actionDim = env.actionSpace.shape[0]
policy = Actor(relstateDim, actionDim)
policy.loadParameters(ADP_dir)


# stateAdp = env.resetRandom(1, noise = 1, MPCtest = True) # [v,omega,x,y,phi,xr,yr,phir]
# relState = env.relStateCal(stateAdp) # [v, omega, dL, dphi]
# .onnx
# Model_save = ADP_dir + '/policy_model.onnx'
# torch_out = policy(relState)
# # Export the model
# torch.onnx.export(policy,               # model being run
#                   relState,                         # model input (or a tuple for multiple inputs)
#                   Model_save,   # where to save the model (can be a file or file-like object)
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=10,          # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names = ['input'],   # the model's input names
#                   output_names = ['output'], # the model's output names
#                   dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
#                                 'output' : {0 : 'batch_size'}})
# onnx_model = onnx.load(Model_save)
# onnx.checker.check_model(onnx_model)
# print(onnx.helper.printable_graph(onnx_model.graph)) # 打印计算图


# .pt
Model_save = ADP_dir + '/policy_model.pt'
stateAdp = env.resetRandom(1, noise = 1, MPCtest = True) # [v,omega,x,y,phi,xr,yr,phir]
relState = env.relStateCal(stateAdp) # [v, omega, dL, dphi]
torch.jit.trace(policy, relState).save(Model_save)


# compare
# policy2 = Actor(relstateDim, actionDim)
# policy2.loadParameters('./Results_dir/2022-04-13-17-31-23')
# for i in range(5):
#     stateAdp = env.resetRandom(1, noise = 0.5, MPCtest = True) # [v,omega,x,y,phi,xr,yr,phir]
#     relState = env.relStateCal(stateAdp) # [v, omega, dL, dphi]
#     print('-' * 50)
#     print('relState: {}'.format(relState[0].tolist()))
#     print('policy 1: {}'.format(policy(relState)[0].tolist()))
#     print('policy 2: {}'.format(policy2(relState)[0].tolist()))

