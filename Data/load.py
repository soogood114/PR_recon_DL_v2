import os
import numpy as np
import glob
from random import shuffle

import Data.exr as exr
import Data.manipulate as mani
import Data.normalization as norm
import Data.design_matrix as design

"""*******************************************************************************"""
"load.py에 대한 정보"
" network train and test에 앞서 exr 또는 numpy 형태의 dataset을 SSD에서 불러오는 함수들로 구성"
"""*******************************************************************************"""


def get_exr_dataset(dataset_dir, suffix='*.exr'):
    dataset = []
    files = os.listdir(dataset_dir)

    files = [fn for fn in glob.glob(os.path.join(dataset_dir, suffix))]

    for file in files:
        filename = os.path.join(dataset_dir, file)
        # print("Loading: ", filename)
        data = exr.read(filename)
        # data = np.clip(data, 0, 1)
        dataset.append(data[:, :, 0:3])

    dataset = np.array(dataset)

    return dataset


def get_dat_dataset(dataset_dir, exr_dataset):
    dataset = []
    files = os.listdir(dataset_dir)

    for i, file in enumerate(files):
        filename = os.path.join(dataset_dir, file)
        data = np.fromfile(filename, dtype=np.float32)

        h, w, _ = exr_dataset[i].shape
        data = np.reshape(data, (h, w, 3))

        dataset.append(data)

    dataset = np.array(dataset)
    return dataset


def save_all_exr_dataset(dataset_dirs, scene, target):
    all_data = []
    for dataset_dir in dataset_dirs:
        files = os.listdir(dataset_dir)
        files = [fn for fn in glob.glob(os.path.join(dataset_dir, '*.exr'))]

        for f in files:
            filename = os.path.join(dataset_dir, f)
            data = exr.read_all(filename)
            all_data.append(data['default'][:, :, 0:3])
            exr.write(os.path.join('D:/training/', target, scene, f), data['default'][:, :, 0:3])

    return np.array(all_data)


def get_all_exr_dataset(dataset_dirs, suffix=""):
    all_data = []
    for dataset_dir in dataset_dirs:
        files = os.listdir(dataset_dir)
        files = [file for file in files if file.endswith(".exr")]
        # files = [fn for fn in glob.glob(os.path.join(dataset_dir, suffix))]

        for f in files:
            filename = os.path.join(dataset_dir, f)
            data = exr.read_all(filename)
            all_data.append(data['default'][:, :, 0:3])

    return np.array(all_data)


def get_all_npy_dataset(dataset_dirs):
    all_data = []
    for dataset_dir in dataset_dirs:
        files = os.listdir(dataset_dir)
        files = [file for file in files if file.endswith(".npy")]
        for f in files:
            filename = os.path.join(dataset_dir, f)
            data = np.load(filename)
            all_data.append(data)

    return np.array(all_data)


def get_all_exr_dataset_one(dataset_dirs, suffix=""):
    all_data = []
    for dataset_dir in dataset_dirs:
        files = os.listdir(dataset_dir)
        files = [file for file in files if file.endswith(".exr")]
        # files = [fn for fn in glob.glob(os.path.join(dataset_dir, suffix))]

        filename = os.path.join(dataset_dir, files[0])
        data = exr.read_all(filename)
        all_data.append(data['default'][:, :, 0:3])

    return np.array(all_data)


def get_all_npy_dataset_one(dataset_dirs):
    all_data = []
    for dataset_dir in dataset_dirs:
        files = os.listdir(dataset_dir)
        files = [file for file in files if file.endswith(".npy")]

        filename = os.path.join(dataset_dir, files[0])
        data = np.load(filename)
        all_data.append(data)

    return np.array(all_data)



def shuffle_list(*ls):
    l = list(zip(*ls))
    shuffle(l)
    return zip(*l)


def concatenate_4d_dataset(*datasets):
    dataset_group = list(zip(*datasets))
    merge = [np.concatenate(ele, axis=2) for ele in dataset_group]
    return np.array(merge)


def get_bmp_dataset_pair(dataset_dir):
    from scipy import misc

    resized_dataset = []
    dataset = []

    scale = 2
    files = os.listdir(dataset_dir)

    for file in files:
        filename = os.path.join(dataset_dir, file)
        data = misc.imread(filename)

        resized_data = misc.imresize(data, size=1.0 / scale, interp='bicubic')
        resized_data = misc.imresize(resized_data, size=scale * 100, interp='bicubic')

        dataset.append(data[:, :, 0:3])
        resized_dataset.append(resized_data[:, :, 0:3])

        exr.write('debug-bmp/' + file + '.exr', data[:, :, 0:3] / 255.0)
        exr.write('debug-bmp-resized/' + file + '.exr', resized_data[:, :, 0:3] / 255.0)

    resized_dataset = np.array(resized_dataset)
    dataset = np.array(dataset)

    return resized_dataset, dataset



"""new ones"""

def get_all_stack_npy_for_each_buffer(dataset_dirs, common_name="total_s_4_", ref_pt=True):
    """
    input : dataset dir pth
    output : throughput, direct, g_buffer, GT
    feature #1 : 각 buffer에 따라 출력을 냄.
    """
    all_throughput_stack = np.load(dataset_dirs + common_name + "throughput_stack.npy")
    all_direct_stack = np.load(dataset_dirs + common_name + "direct_stack.npy")
    all_g_buffer_stack = np.load(dataset_dirs + common_name + "g_buffer_stack.npy")
    if ref_pt:
        all_GT_stack = np.load(dataset_dirs + common_name + "GT_stack_pt.npy")
    else:
        all_GT_stack = np.load(dataset_dirs + common_name + "GT_stack_weighted.npy")

    return all_throughput_stack, all_direct_stack, all_g_buffer_stack, all_GT_stack

def get_all_stack_npy_for_input_buffer(dataset_dirs, common_name="total_s_4_", ref_pt=True):
    """
    input : dataset dir pth
    output : input_stack(throughput + direct, g_buffer), GT
    feature #1 : input stack을 위와 같이 만드러 출력, throughput은 direct와 합쳐짐.
    """
    all_throughput_stack = np.load(dataset_dirs + common_name + "throughput_stack.npy")
    all_direct_stack = np.load(dataset_dirs + common_name + "direct_stack.npy")
    all_g_buffer_stack = np.load(dataset_dirs + common_name + "g_buffer_stack.npy")
    if ref_pt:
        all_GT_stack = np.load(dataset_dirs + common_name + "GT_stack_pt.npy")
    else:
        all_GT_stack = np.load(dataset_dirs + common_name + "GT_stack_weighted.npy")


    "debug"
    # throughput_stack_test = all_throughput_stack[0, :, :, 0, :]
    # exr.write("./throughput_stack_test.exr", throughput_stack_test)
    #
    # GT_stack_test = all_GT_stack[0, :, :, 0, :]
    # exr.write("./GT_stack_test.exr", GT_stack_test)
    #
    # direct_stack_test = all_direct_stack[0, :, :, 0, :]
    # exr.write("./direct_stack_test.exr", direct_stack_test)
    #
    # albedo_stack_test = all_g_buffer_stack[0, :, :, 0, :3]
    # exr.write("./albedo_stack_test.exr", albedo_stack_test)
    #
    # depth_stack_test = all_g_buffer_stack[0, :, :, 0, 3]
    # exr.write("./depth_stack_test.exr", depth_stack_test)

    return np.concatenate((all_throughput_stack + all_direct_stack, all_g_buffer_stack), axis=4), all_GT_stack



def get_all_img_npy_for_input_buffer(dataset_dirs, common_name="total_s_4_"):
    """
    input : dataset dir pth
    output : input_img(throughput + direct, g_buffer), GT
    feature #1 : 위의 함수와는 다르게 stack이 아니라, img 형태로 출력
    """
    all_throughput_img = mani.make_full_res_img_numpy(np.load(dataset_dirs + common_name + "throughput_stack.npy"))
    all_direct_img = mani.make_full_res_img_numpy(np.load(dataset_dirs + common_name + "direct_stack.npy"))
    all_g_buffer_img = mani.make_full_res_img_numpy(np.load(dataset_dirs + common_name + "g_buffer_stack.npy"))
    all_GT_img = mani.make_full_res_img_numpy(np.load(dataset_dirs + common_name + "GT_stack_weighted.npy"))

    return np.concatenate((all_throughput_img + all_direct_img, all_g_buffer_img), axis=3), all_GT_img




def get_input_design_stack_and_normalize(dirs, common_name, params):
    """
    메모리 효율을 위해 이 함수 내에서 loading과 normalization 둘다를 하도록 함.
    또한, input과 design에서 서로
    """
    # load
    input_stack, GT_stack = get_all_stack_npy_for_input_buffer(dirs, common_name, params['ref_pt'])

    # normalization
    norm.normalize_input_stack_v1(input_stack)
    norm.normalize_GT_v1(GT_stack)

    # design matrix
    design_stack = design.generate_design_mat_from_stack_v1(input_stack[:, :, :, :, 3:],
                                                            params['tile_length'], params['grid_order'])

    # exclude boundary if it is ok
    s = params['tile_length']
    tile_size = s ** 2
    tile_size_stit = s ** 2 + s * 2

    if params["no_boundary_for_input"]:
        ch_input = tile_size
    else:
        ch_input = tile_size_stit


    if params["no_boundary_for_design"]:
        # design은 나중에 나올 output의 형태를 결정. 따라서 loss를 구하기 위해 GT도 그에 맞춰야 함.
        ch_design = tile_size
        ch_gt = tile_size
    else:
        ch_design = tile_size_stit
        ch_gt = tile_size_stit

    return input_stack[:, :, :, :ch_input, :], design_stack[:, :, :, :ch_design, :], GT_stack[:, :, :, :ch_gt, :]

