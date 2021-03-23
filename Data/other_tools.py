import numpy as np
import torch
import Models.net_op as net_op

def calcRelMSE(input, ref):
    h, w, _ = np.shape(input)
    num = np.square(np.subtract(input, ref))
    denom = np.mean(ref, axis=2)
    denom = np.reshape(denom, (h, w, 1))
    relMSE = num / (denom * denom + 1e-2)
    relMSEMean = np.mean(relMSE)
    return relMSEMean


def from_torch_tensor_stack_to_full_res_numpy(torch_tensor_stack):
    """
    input : stack version of torch tensor (b tile_size c h_d w_d)
    output : full res version of numpy tensor
    """

    torch_tensor = net_op.make_full_res_img_torch(torch_tensor_stack)  # b c h w
    torch_tensor_np = torch_tensor.cpu().detach().numpy()
    torch_tensor_np = np.transpose(torch_tensor_np, (0, 2, 3, 1))

    return torch_tensor_np

