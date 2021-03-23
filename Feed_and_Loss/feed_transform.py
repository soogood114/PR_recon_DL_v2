import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data
import Data.exr as exr

"""*******************************************************************************"""
"feed_transform.py에 대한 정보"
" network feed를 하기 앞서 patch를 만들고 transform을 하여 torch tensor를 만드는 함수들로 구성"
" 즉, 목저은 patch generation, data augmentation, torch tensor generation 이다."
"""*******************************************************************************"""


class RandomCrop_stack_with_design(object):
    """
    input : output_size, tile_size, stack_img_sample(input, design, GT)
    output : stack_patch_sample
    feature : 한 마디로 patch를 random하게 각 이미지 마다 한 장을 만들어 내는 용도임.
    """

    def __init__(self, output_size, tile_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.tile_size = tile_size

    def __call__(self, sample):
        input, design, target = sample['input'], sample['design'], sample['target']

        h_d, w_d = input.shape[:2]
        new_h_d, new_w_d = self.output_size

        new_h_d //= self.tile_size
        new_w_d //= self.tile_size

        top = np.random.randint(0, h_d - new_h_d)
        left = np.random.randint(0, w_d - new_w_d)

        input = input[top: top + new_h_d,
                left: left + new_w_d]

        design = design[top: top + new_h_d,
                left: left + new_w_d]

        target = target[top: top + new_h_d,
                 left: left + new_w_d]

        return {'input': input, 'design': design, 'target': target}


class RandomCrop_img_stack_with_design(object):
    """
    input : output_size, tile_size, stack_img_sample(input, design, GT)
    output : stack_patch_sample
    feature #1 : 한 마디로 patch를 random하게 각 이미지 마다 한 장을 만들어 내는 용도임.
    feature #2 : img형태의 input과 stack 형태의 design을 대상으로 함.
    feature #3 : 한마디로 img와 stack 모두를 다룸
    """

    def __init__(self, output_size, tile_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.tile_size = tile_size

    def __call__(self, sample):
        input, design, target = sample['input'], sample['design'], sample['target']

        h_d, w_d = input.shape[:2]
        new_h, new_w = self.output_size

        new_h_d = new_h // self.tile_size
        new_w_d = new_w // self.tile_size

        top_d = np.random.randint(0, h_d - new_h_d)
        left_d = np.random.randint(0, w_d - new_w_d)

        top = top_d * self.tile_size
        left = left_d * self.tile_size

        input = input[top: top + new_h,
                left: left + new_w]

        target = target[top: top + new_h,
                left: left + new_w]

        design = design[top_d: top_d + new_h_d,
                 left_d: left_d + new_w_d]

        return {'input': input, 'design': design, 'target': target}




class RandomFlip_with_design(object):
    """
    input : patch_img_sample(input, design, GT)
    output : rotated patch_img_sample(input, design, GT) according to h and w
    feature #1 : h w 각각에 대해 50% 확률로 뒤집어 놓음.
    !!! 현재 문제가 있음을 확인
    """

    def __init__(self, multi_crop=False):
        self.multi_crop = multi_crop

    def __call__(self, sample):
        input, design, target = sample['input'], sample['design'], sample['target']
        # H, W, C

        prob_vertical_flip = np.random.randint(0, 100)
        prob_horizontal_flip = np.random.randint(0, 100)

        if not self.multi_crop:
            axis_index = 0
        else:
            axis_index = 1

        if prob_vertical_flip > 50:
            input = np.flip(input, axis=axis_index)
            design = np.flip(design, axis=axis_index)
            target = np.flip(target, axis=axis_index)

        if prob_horizontal_flip > 50:
            input = np.flip(input, axis=axis_index + 1)
            design = np.flip(design, axis=axis_index + 1)
            target = np.flip(target, axis=axis_index + 1)

        return {'input': input.copy(), 'design': design.copy(), 'target': target.copy()}


class ToTensor_stack_with_design(object):
    """
    input : numpy patch_img_sample(input, design, GT)
    output : torch tensor patch_img_sample(input, design, GT)
    feature #1 : 단순히 numpy patch를 torch tensor로 바꿈.
    feature #2 : torch tensor에 맞춰 N C H W 형태로 바꾸고 아직은 GPU에 올라가질 않음.
    """

    def __init__(self, multi_crop=False, double_to_float=True):
        self.multi_crop = multi_crop
        self.double_to_float = double_to_float

    def __call__(self, sample):
        input, design, target = sample['input'], sample['design'], sample['target']

        # input : (multi_crop) h w  tile_size c
        # target : (multi_crop) h w  tile_size c

        # swap color axis because
        # numpy image: H x W x C or N x H x W x C
        # torch image: C X H X W or N x C X H X W

        if not self.multi_crop:
            input = input.transpose((2, 3, 0, 1))
            design = design.transpose((2, 3, 0, 1))
            target = target.transpose((2, 3, 0, 1))
        else:
            input = input.transpose((0, 3, 4, 1, 2))
            design = design.transpose((0, 3, 4, 1, 2))
            target = target.transpose((0, 3, 4, 1, 2))

        return {'input': torch.from_numpy(input).to(dtype=torch.float),
                'design': torch.from_numpy(design).to(dtype=torch.float),
                'target': torch.from_numpy(target).to(dtype=torch.float)}

class ToTensor_img_stack_with_design(object):
    """
    input : numpy patch_img_sample(input, design, GT)
    output : torch tensor patch_img_sample(input, design, GT)
    feature #1 : 단순히 numpy patch를 torch tensor로 바꿈.
    feature #2 : torch tensor에 맞춰 N C H W 형태로 바꾸고 아직은 GPU에 올라가질 않음.
    feature #3 : img형태의 input과 stack 형태의 design을 대상으로 함.
    feature #4 : 한마디로 img와 stack 모두를 다룸
    """

    def __init__(self, multi_crop=False, double_to_float=True):
        self.multi_crop = multi_crop
        self.double_to_float = double_to_float

    def __call__(self, sample):
        input, design, target = sample['input'], sample['design'], sample['target']

        # input : (multi_crop) h w  tile_size c
        # target : (multi_crop) h w  tile_size c

        # swap color axis because
        # numpy image: H x W x C or N x H x W x C
        # torch image: C X H X W or N x C X H X W

        if not self.multi_crop:
            input = input.transpose((2, 0, 1))
            design = design.transpose((2, 3, 0, 1))
            target = target.transpose((2, 0, 1))
        else:
            input = input.transpose((0, 3, 1, 2))
            design = design.transpose((0, 3, 4, 1, 2))
            target = target.transpose((0, 3, 1, 2))

        return {'input': torch.from_numpy(input).to(dtype=torch.float),
                'design': torch.from_numpy(design).to(dtype=torch.float),
                'target': torch.from_numpy(target).to(dtype=torch.float)}








