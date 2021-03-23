import numpy as np
import Data.manipulate as mani

"""*******************************************************************************"""
"design_matrix.py에 대한 정보"
" torch tensor 단에서 이뤄진 design matrix 생성을 netwrok feed에 앞서 미리 구현."
" 이름 그대로 numpy 단에서 design matrix를 만들고 그에 필요한 함수들로 구성"
"""*******************************************************************************"""

def full_res_grid_stack_for_tensor_npy(num_imgs, img_height, img_width, tile_length, start=0, end=1, no_boundary=True):
    """
    input : 텐서의 크기 정보와 grid의 범위
    output : 여러 이미지 텐서를 상대로 이미지 크기 형태의 x, y grid를 [start, end]의 범위로 생성
    feature #1 : output 또한 stack version으로 생성이 됨
    feature #2 : 이름에서 나왔듯이 stack 버전의 형태로 출력이 나옴.
    feature #3 : 또한, stitching boundary 여부를 결정할 수 있음.
    """
    s = int(tile_length)
    tile_size = s ** 2
    tile_size_stit = s ** 2 + 2 * s
    if no_boundary:
        ch = tile_size
    else:
        ch = tile_size_stit
    out = np.zeros((num_imgs, img_height // s, img_width // s, ch, 2), dtype=float)

    xx = np.linspace(start, end, img_width)
    yy = np.linspace(start, end, img_height)
    grid_x, grid_y = np.meshgrid(xx, yy)
    grid = np.concatenate((np.expand_dims(grid_x, axis=(0, 3)), np.expand_dims(grid_y, axis=(0, 3))), axis=3)
    grid = np.repeat(grid, num_imgs, axis=0)

    dx_grid, dy_grid = mani.CalcGrad_tensor(grid)

    grid_x_shift = grid + dx_grid
    grid_y_shift = grid + dy_grid

    for index in range(tile_size):
        i = index // s
        j = index % s

        out[:, :, :, j + s * i, :] = grid[:, i::s, j::s, :]

    if not no_boundary:
        for i in range(s):
            # x shift
            out[:, :, :, tile_size + i, :] = grid_x_shift[:, i::s, (s - 1)::s, :]

            # y shift
            out[:, :, :, tile_size + s + i, :] = grid_y_shift[:, (s - 1)::s, i::s, :]

    return out


def full_res_grid_img_for_tensor_npy(num_imgs, img_height, img_width, start=0, end=1):
    """
    input : 텐서의 크기 정보와 grid의 범위
    output : 여러 이미지 텐서를 상대로 이미지 크기 형태의 x, y grid를 [start, end]의 범위로 생성
    feature : output 또한 full res image 형태로 생성이 됨
    """

    xx = np.linspace(start, end, img_width)
    yy = np.linspace(start, end, img_height)
    grid_x, grid_y = np.meshgrid(xx, yy)
    grid = np.concatenate((np.expand_dims(grid_x, axis=(0, 3)), np.expand_dims(grid_y, axis=(0, 3))), axis=3)
    grid = np.repeat(grid, num_imgs, axis=0)
    return grid


def generate_design_mat_from_stack_v1(g_buffer_stack, tile_length, grid_order=2, grid_start=0, grid_end=1,
                                      no_boundary_for_design=False):
    """
    input : g_buffer_stack
    output : design_mat with stack version
    feature #1 : 기존에 pytorch 함수에서 한 것이랑 비슷. 속도와 메모리를 위해서 따로 뺌.
    feature #2 : stack version에 따름.
    """
    N, H_d, W_d, _, C = g_buffer_stack.shape
    s = tile_length
    H = int(s * H_d)
    W = int(s * W_d)


    tile_size = s**2
    tile_size_stit = s**2 + 2*s

    if no_boundary_for_design:
        design_mat = np.ones((N, H_d, W_d, tile_size, C + 1 + grid_order * 2), dtype=g_buffer_stack.dtype)
    else:
        design_mat = np.ones((N, H_d, W_d, tile_size_stit, C + 1 + grid_order * 2), dtype=g_buffer_stack.dtype)


    full_res_grid_stack = full_res_grid_stack_for_tensor_npy(N, H, W, tile_length, grid_start, grid_end,
                                                             no_boundary_for_design)

    # design_mat = np.concatenate((ones_stack, g_buffer_stack), axis=4)

    design_mat[:, :, :, :, 1:1 + C] = g_buffer_stack
    # del g_buffer_stack

    # include grid
    index_start = 1 + C
    for i in range(grid_order):
        order = i + 1
        design_mat[:, :, :, :, index_start:index_start + 2] = np.power(full_res_grid_stack, order)
        index_start += 2

    return design_mat



def generate_design_mat_from_img_v1(g_buffer_img, tile_length, grid_order=2, grid_start=0, grid_end=1,
                                    design_for_stit=False):
    """
    input : g_buffer_image
    output : design_mat with stack version
    feature #1 : design은 무조건 stack 버전이 되어야 한다. 이는 unfold 때문.
    feature #2 : 그런데 이 함수의 입력은 img 형태의 데이터가 된다.
    feature #3 : 여기선 boundary를 신경 쓰지 않음. no stitching !
    """
    N, H, W, C = g_buffer_img.shape
    s = tile_length
    H_d = int(H // s)
    W_d = int(W // s)
    tile_size = s ** 2
    tile_size_stit = s ** 2 + s * 2

    if design_for_stit:
        design_mat = np.ones((N, H_d, W_d, tile_size_stit, C + 1 + grid_order * 2), dtype=g_buffer_img.dtype)
        g_buffer_stack = mani.stack_elements_general_version(g_buffer_img)
    else:
        design_mat = np.ones((N, H_d, W_d, tile_size, C + 1 + grid_order * 2), dtype=g_buffer_img.dtype)
        g_buffer_stack = mani.stack_elements_general_version_no_stitching(g_buffer_img)

    full_res_grid_stack = full_res_grid_stack_for_tensor_npy(N, H, W, tile_length, grid_start, grid_end, design_for_stit)

    design_mat[:, :, :, :, 1:1 + C] = g_buffer_stack
    del g_buffer_stack

    # include grid
    index_start = 1 + C
    for i in range(grid_order):
        order = i + 1
        design_mat[:, :, :, :, index_start:index_start + 2] = np.power(full_res_grid_stack, order)
        index_start += 2

    return design_mat
