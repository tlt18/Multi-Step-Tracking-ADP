from myenv import TrackingEnv
from network import Actor, Critic, ActorForIDC
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np

ADP_dir = './Results_dir/2023-11-09-13-19-20'
env = TrackingEnv()
relstateDim = env.relstateDim
actionDim = env.actionSpace.shape[0]
policy = ActorForIDC(relstateDim, actionDim)
policy.loadParameters(ADP_dir)

stateAdp = env.resetRandom(1, noise = 1, MPCflag = 0) # [v,omega,x,y,phi,xr,yr,phir]
relState = env.relStateCal(stateAdp) # [v, omega, dL, dphi]

# stateForIDC = torch.zeros([1, 122])
stateForIDC = torch.rand([1, 122])
# .onnx
Model_save = ADP_dir + '/policy_tlt.onnx'
torch_out = policy(stateForIDC)
# Export the model
torch.onnx.export(policy,               # model being run
                  stateForIDC,                         # model input (or a tuple for multiple inputs)
                  Model_save,   # where to save the model (can be a file or file-like object)
                  input_names = ['input'],
                  output_names = ['output'], # the model's output names
                  opset_version=11,          # the ONNX version to export the model to
                )

# check onnx model with ONNX API
onnx_model = onnx.load(Model_save)
onnx.checker.check_model(onnx_model)
# print(onnx.helper.printable_graph(onnx_model.graph)) # 打印计算图

# compute output
ort_session = ort.InferenceSession(Model_save)
ort_inputs = {ort_session.get_inputs()[0].name: stateForIDC.numpy()}
ort_outs = ort_session.run(None, ort_inputs)

print("torch_out: {}".format(torch_out.detach().numpy()))
print("ort_outs: {}".format(ort_outs[0]))

np.testing.assert_allclose(torch_out.detach().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)

# .pt
# Model_save = ADP_dir + '/policy_model.pt'
# stateAdp = env.resetRandom(1, noise = 1, MPCtest = True) # [v,omega,x,y,phi,xr,yr,phir]
# relState = env.relStateCal(stateAdp) # [v, omega, dL, dphi]
# torch.jit.trace(policy, relState).save(Model_save)


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

