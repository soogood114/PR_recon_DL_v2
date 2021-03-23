import numpy as np
from numba import jit
import math
import Data.exr as exr

"""*******************************************************************************"""
"manipulate.py에 대한 정보"
" numpy 형태의 데이터를 필요에 따라 transform하는 형태의 함수 구성 "
" ex) stack, full_res, gradient etc."
" 단, loss와 feed에 대한 numpy 연산은 여기에 없고 다른 폴더에 있음."
"""*******************************************************************************"""

def CalcGrad_tensor(data):

    s, h, w, c = data.shape
    dX = data[:, :, 1:, :] - data[:, :, :w - 1, :]
    dY = data[:, 1:, :, :] - data[:, :h - 1, :, :]
    dX = np.concatenate((dX, np.zeros([s, h,1,c])), axis=2)
    dY = np.concatenate((dY, np.zeros([s, 1,w,c])), axis=1)
    return dX,dY

def CalcGrad(data):

    h, w, c = data.shape
    dX = data[:, 1:, :] - data[:, :w - 1, :]
    dY = data[1:, :, :] - data[:h - 1, :, :]
    dX = np.concatenate((dX, np.zeros([h,1,c])), axis=1)
    dY = np.concatenate((dY, np.zeros([1,w,c])), axis=0)
    return dX,dY

def stack_elements_general_version(input_tensor, s):
    """
    input : 입격으로 들어오는 tensor단위의 이미지
    output : stack version with stitching
    feature : 기본으로 stitching의 정보를 담고 있음.
    """
    N, H, W, C = input_tensor.shape

    tile_size = s ** 2
    tile_size_stit = s ** 2 + s * 2  # 16 + 8 = 24

    out = np.zeros((N, H // s, W // s, tile_size_stit, C), dtype=input_tensor.dtype)

    tile_tensor = input_tensor  # tile_img, 3ch
    grad_dx_tensor, grad_dy_tensor = CalcGrad_tensor(input_tensor)

    tile_tensor_x_sh = tile_tensor + grad_dx_tensor  # 3ch
    tile_tensor_y_sh = tile_tensor + grad_dy_tensor  # 3ch

    del grad_dx_tensor
    del grad_dy_tensor

    # tile
    for index in range(tile_size):
        i = index // s
        j = index % s

        out[:, :, :, j + s * i, :] = tile_tensor[:, i::s, j::s, :]

    for i in range(s):
        # x shift
        out[:, :, :, tile_size + i, :] = tile_tensor_x_sh[:, i::s, (s - 1)::s, :]

        # y shift
        out[:, :, :, tile_size + s + i, :] = tile_tensor_y_sh[:, (s - 1)::s, i::s, :]

    return out


def stack_elements_general_version_no_stitching(input_tensor, s):
    """
    input : 입격으로 들어오는 tensor단위의 이미지
    output : stack version with stitching
    feature : 이름에서 나왔듯이 stitching에 대한 정보는 안 담음.
    """
    N, H, W, C = input_tensor.shape

    tile_size = s ** 2

    out = np.zeros((N, H // s, W // s, tile_size, C), dtype=input_tensor.dtype)

    # tile
    for index in range(tile_size):
        i = index // s
        j = index % s

        out[:, :, :, j + s * i, :] = input_tensor[:, i::s, j::s, :]

    return out


def make_full_res_img_numpy(out_stack, tile_length=4):

    b, h, w, _, c = out_stack.shape

    # out_stack : (b, 3, h // s, w // s, tile_size)
    s = int(tile_length)

    full_res_img = np.zeros((b, h * s, w * s, c), dtype=out_stack.dtype)

    # for i in range(s):
    #     for j in range(s):
    #         full_res_img[:, :, i::s, j::s] = out_stack[:, :, :, :, j + s * i]

    for index in range(s ** 2):
        i = index // s
        j = index % s
        full_res_img[:, i::s, j::s, :] = out_stack[:, :, :,  j + s * i, :]

    return full_res_img