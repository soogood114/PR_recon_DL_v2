import os
import numpy as np
import pickle
import Data.exr as exr

"""*******************************************************************************"""
"normalization.py에 대한 정보"
" 학습에 안정을 위한 정규확 함수들을 numpy 데이터 상대로 구성해 놓음."
"""*******************************************************************************"""


def normalization_signed_log(data):
    """
    특징 : 원래 하던대로 하는 방법, inplacde 함수
    """
    return np.sign(data) * np.log(np.absolute(data) + 1)


def denormalization_signed_log(data):
    """
    특징 : log transform의 역연산
    """
    return np.sign(data) * (np.exp(np.absolute(data)) - 1)

def normalize_depth_1ch_v1(depth):
    """
    input : N, H_d, W_d, T,  - > 여기서는 3개의 채널, stack version
    output : 각 depth 이미지 마다 최대값을 나눠 [0, 1]로 만듦.
    특징 : stack과 image 둘 버전의 depth가 쓰임.
    """
    n_data = depth.shape[0]
    output = np.zeros_like(depth)
    for i in range(n_data):
        max_depth = np.max(depth[i])
        output[i] = depth[i] / max_depth
    return output

def normalize_g_buffer_v1(g_buffer):
    """
    input : N, H_d, W_d, T, 7 : 7은 albedo depth normal 순서
    output : albedo = 이미 [0, 1], depth = 1ch [0, 1], normal = [0, 1]
    특징 : 원래 하던대로 하는 방법, inplacde 함수
    """
    # depth
    g_buffer[:, :, :, :, 3] = normalize_depth_1ch_v1(g_buffer[:, :, :, :, 3])

    # normal
    g_buffer[:, :, :, :, 4:] += 1
    g_buffer[:, :, :, :, 4:] /= 2

    return g_buffer


def normalize_input_stack_v1(input):
    """
        input : N, H_d, W_d, T, 10 : 7은 albedo depth normal 순서
        output : throughput log ,albedo = 이미 [0, 1], depth = 1ch [0, 1], normal = [0, 1]
        특징 : input을 위한 최적의 연산을 위해 위의 함수를 이용
    """
    # exr.write("./_throughput_stack_test.exr", input[0, :, :, 0, :3])
    # exr.write("./_depth_stack_test.exr", input[0, :, :, 0, 6])
    # exr.write("./_normal_stack_test.exr", input[0, :, :, 0, 7:])

    # color log transform
    input[:, :, :, :, :3] = normalization_signed_log(input[:, :, :, :, :3])

    # albedo skip

    # depth
    input[:, :, :, :, 6] = normalize_depth_1ch_v1(input[:, :, :, :, 6])

    # normal
    input[:, :, :, :, 7:] += 1
    input[:, :, :, :, 7:] /= 2


def normalize_input_img_v1(input):
    """
        input : N, H, W, 10 : 7은 albedo depth normal 순서
        output : throughput log ,albedo = 이미 [0, 1], depth = 1ch [0, 1], normal = [0, 1]
        특징 : 입력을 stack의 형태가 아니라 image의 형태로 받게 된다.
    """
    # exr.write("./_throughput_stack_test.exr", input[0, :, :, 0, :3])
    # exr.write("./_depth_stack_test.exr", input[0, :, :, 0, 6])
    # exr.write("./_normal_stack_test.exr", input[0, :, :, 0, 7:])

    # color log transform
    input[:, :, :, :3] = normalization_signed_log(input[:, :, :, :3])

    # albedo skip

    # depth
    input[:, :, :, 6] = normalize_depth_1ch_v1(input[:, :, :, 6])

    # normal
    input[:, :, :, 7:] += 1
    input[:, :, :, 7:] /= 2


def normalize_GT_v1(GT):
    GT[:, :, :, :, :3] = normalization_signed_log(GT[:, :, :, :, :3])

