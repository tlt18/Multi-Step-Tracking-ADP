from myenv import TrackingEnv
from network import Actor, Critic
import torch.onnx
import onnx

ADP_dir = './Results_dir/2022-05-11-20-48-35'

env = TrackingEnv()
relstateDim = env.relstateDim
actionDim = env.actionSpace.shape[0]
policy = Actor(relstateDim, actionDim)
policy.loadParameters(ADP_dir)

stateAdp = env.resetRandom(1, noise = 1, MPCtest = True) # [v,omega,x,y,phi,xr,yr,phir]
relState = env.relStateCal(stateAdp) # [v, omega, dL, dphi]

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