import os
import numpy as np
import json
from datetime import datetime

import Data.load as load
import V_cmp.ttv_Vcmp as ttv

params_default = {
    # 1. Mode selection
    "mini_batch": True,  # mini batch

    'trained_model_test': False,  # test mode
    'trained_parameter_pth': "210219_1500_epoch",
    'trained_parameter_name': "latest_parameter",

    # 2. Data load
    'ref_pt': False,
    'use_noisy_color': False,
    'use_gradients': False,

    # 4. Image and batch size & iterations
    'batch_size': 6,  # 32  배치사이즈가 지나치게 크게 되면 자동으로 잘라준다.
    'epochs': 10,
    'patch_size': 200,  # 200
    'multi_crop': False,

    # 5. Normalization
    'mue_tr': False,

    # 6. Loss configuration
    'loss_type': 'l1',  # l1, l2, smape

    # 7. Optimization
    'optim': 'adam',
    'lr': 0.0001,  # default : 0.0001

    # 8. Saving period
    "para_saving_epoch": 100,  # 100
    "loss_saving_epoch": 10,  # 10
    "val_patches_saving_epoch": 100,  # 100

    # 9. Index setting for run and network functions
    'run_index': 0,
    'network_index': 1,
    'time_saving_folder': "tmp",  # it will be made soon
    'saving_folder_name': "210331_NGPT_ttt",  # 210319_model_stack_v2_epoch_2k_norm, 210325_NGPT_epoch_2k
    'saving_file_name': "210331_NGPT_ttt",

}


def data_load_and_run(params=None, gpu_id=1):
    if params is None:
        params = params_default

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    """####################  CREATE SAVING FOLDER ####################"""
    # must make the root of saving results if it is not exist
    root_saving_pth = "./results"
    if not os.path.exists(root_saving_pth):
        os.mkdir(root_saving_pth)

    # make the new saving folder as intended
    saving_folder_pth = root_saving_pth + "/" + params["saving_folder_name"]
    if not os.path.exists(saving_folder_pth):
        os.mkdir(saving_folder_pth)

    # for multiple experiments with same params, time_folder is needed.
    params["time_saving_folder"] = saving_folder_pth + "/" + str(datetime.today().strftime("%m_%d_%H_%M_%S"))
    if not os.path.exists(params["time_saving_folder"]):
        os.mkdir(params["time_saving_folder"])

    # saving folder -> tile- > tilme foldoer -> setting, img, tensorboard ect.

    """####################  SAVE THE SETTING ####################"""
    # saving folder -> tile- > tilme foldoer -> setting, img, tensorboard ect.
    json_saving_folder_pth = params["time_saving_folder"] + "/settings"
    if not os.path.exists(json_saving_folder_pth):
        os.mkdir(json_saving_folder_pth)

    json_saving_pth = json_saving_folder_pth + "/setting_params_for_cmp.json"
    with open(json_saving_pth, 'w') as fp:
        json.dump(params, fp)

    if not params["trained_model_test"]:

        """####################  TRAIN MODE ####################"""
        dataset_dirs = "C:/DB_FOR_SIGA21/"

        # 원래 ttv에서 나온 buffer가 아니라서 normalization이 없음.
        # train_input_img_buffer, train_ref_img_buffer, test_input_img_buffer, test_ref_img_buffer = \
        #     load.get_all_img_exr_for_ttv_v2(dataset_dirs, params['mini_batch'],
        #                                     False, params['use_noisy_color'], params['use_gradients'])

        train_input_img_buffer, train_ref_img_buffer, test_input_img_buffer, test_ref_img_buffer = \
            load.get_all_img_exr_from_stack_npy(dataset_dirs, params['mini_batch'], False)

        ttv.train_test_cmp_model_img_v1(train_input_img_buffer, train_ref_img_buffer, test_input_img_buffer,
                                        test_ref_img_buffer, params)
    else:
        """####################  TEST MODE ####################"""
        dataset_dirs = "C:/DB_FOR_SIGA21/"

        _, _, test_input_img_buffer, test_ref_img_buffer = \
            load.get_all_img_exr_for_ttv_v2(dataset_dirs, params['mini_batch'],
                                            True, params['use_noisy_color'], params['use_gradients'])

        # # 구분을 위해 저장폴더 이름에 다음과 같이 표시를 함.
        # params["saving_folder_name"] = "TEST_" + params["saving_folder_name"]
        #
        #
        # ttv.test_model_stack_v1(test_input_stack, test_design_stack, test_GT_stack, params)
