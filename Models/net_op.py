import torch

def make_full_res_img_torch(out_stack, s=4):
    """
    input : stack version of RGB full res image
    output : image version of RGB full res image
    """
    h, w = out_stack.size(3), out_stack.size(4)
    b = out_stack.size(0)
    c = out_stack.size(2)

    full_res_img = torch.zeros((b, c, h * s, w * s), dtype=out_stack.dtype, layout=out_stack.layout,
                               device=out_stack.device)

    for index in range(s ** 2):
        i = index // s
        j = index % s
        full_res_img[:, :, i::s, j::s] = out_stack[:, j + s * i, :, :, :]

    return full_res_img


