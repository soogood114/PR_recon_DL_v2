import numpy as np
import os
from datetime import datetime
import pickle


import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from tqdm import tqdm as tqdm
from torchvision import transforms, utils
from torch.utils import tensorboard as tensorboard
import torch.optim.lr_scheduler as lr_scheduler

import Data.exr as exr
import Data.other_tools as other_tools

import Data.normalization as norm
import Data.design_matrix as design
import Feed_and_Loss.feed_transform as FT
import Feed_and_Loss.dataset as dataset
import Feed_and_Loss.loss as my_loss

import Models.net_op as net_op
import Models.models_v1 as models_v1


def train_test_model_stack_v1(train_input_stack, train_design_stack, train_GT_stack,
                              test_input_stack, test_design_stack, test_GT_stack, params):
    """
    입력 구성: path reusing recon에 쓰이는 TRAIN AND TEST 버퍼.
    순서: normalization -> making gird by order -> design matrix -> data loader -> network setting -> train -> test
    특징 #1 : 최대한 간단하면서 지적된 문제점을 보안함
    특징 #2 : 일단 무조건 바운더리를 포함함.
    """

    """INITIAL SETTING"""
    # GPU index setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_train, H_d, W_d, tile_size, ch_in = train_input_stack.shape
    N_test = test_input_stack.shape[0]
    H = H_d * params["tile_length"]
    W = W_d * params["tile_length"]


    """SETTING DATA LOAD AND CORRESPONDING TRANSFORMS"""
    # define transform op
    transform_patch = transforms.Compose([
        FT.RandomCrop_stack_with_design(params['patch_size'], params['tile_length']),
        # FT.RandomFlip_with_design(multi_crop=False),  # 현재 문제가 있음.
        FT.ToTensor_stack_with_design(multi_crop=False)
    ])
    transform_img = transforms.Compose([FT.ToTensor_stack_with_design(multi_crop=False)])  # targeting for image

    # train data loader
    train_data = dataset.Supervised_dataset_with_design_v1(train_input_stack, train_design_stack, train_GT_stack,
                                               train=True, transform=transform_patch)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)

    # test data loader
    test_data = dataset.Supervised_dataset_with_design_v1(test_input_stack, test_design_stack, test_GT_stack,
                                                           train=False, transform=transform_img)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)


    """NETWORK INITIALIZATION"""
    # mynet = models_v1.NPR_net_stack_v1(ch_in=10, kernel_size=3, tile_length=4, n_layers=12, length_inter_tile=7,
    #                                    epsilon=1, pad_mode=0, no_stit_input=params['no_boundary_for_input'],
    #                                    no_stit_design=params['no_boundary_for_design']).train().to(device)

    # mynet = models_v1.NPR_net_img_v1(ch_in=10, kernel_size=3, tile_length=4, n_layers=20, length_inter_tile=7,
    #                                  epsilon=0.01, pad_mode=0, no_stit_design=params['no_boundary_for_design']).train().to(device)


    mynet = models_v1.NPR_net_stack_v2(params, ch_in=10, kernel_size=3, tile_length=4, n_layers=12,
                                       length_inter_tile=7, epsilon=0.01, pad_mode=1, unfolded_loss=False,
                                       norm_in_window=True, W_half=False).train().to(device)

    """SAVING THE TENSORBOARD"""
    out_tensorboard_folder_name = params["time_saving_folder"] + "/tensorboards"
    if not os.path.exists(out_tensorboard_folder_name):
        os.mkdir(out_tensorboard_folder_name)
    writer = tensorboard.SummaryWriter(out_tensorboard_folder_name)


    """SET LOSS AND OPTIMIZATION"""
    loss_fn = my_loss.loss_for_stit_v1(params['tile_length'], params["stitching_weights"], params['loss_type'])

    optimizer = optim.Adam(mynet.parameters(), lr=params['lr'])


    """TRAIN NETWORK"""
    epochs = params["epochs"]

    with tqdm(range(0, epochs), leave=True) as tnr:
        tnr.set_postfix(epoch=0, loss=-1.)

        for epoch in tnr:

            one_epoch_loss = 0.0
            num_iter_for_one_epoch = 0

            for data in train_loader:
                optimizer.zero_grad()

                x = data['input'].cuda()
                d = data['design'].cuda()
                y = data['target'].cuda()

                # d_np_saving = other_tools.from_torch_tensor_stack_to_full_res_numpy(d)
                # d_np_saving = d_np_saving[0]
                # exr.write("./train_design_albedo.exr", d_np_saving[:, :, 1:4])
                # exr.write("./train_design_depth.exr", d_np_saving[:, :, 4])
                # exr.write("./train_design_normal.exr", d_np_saving[:, :, 5:8])
                # exr.write("./train_design_xx.exr", d_np_saving[:, :, 8])
                #
                # x_np_saving = other_tools.from_torch_tensor_stack_to_full_res_numpy(x)
                # x_np_saving = x_np_saving[0]
                # exr.write("./train_input_color.exr", x_np_saving[:, :, 0:3])
                #
                # y_np_saving = other_tools.from_torch_tensor_stack_to_full_res_numpy(y)
                # y_np_saving = y_np_saving[0]
                # exr.write("./train_ref_color.exr", y_np_saving[:, :, 0:3])

                # for v1
                # y_pred = mynet(x, d)
                # current_loss = loss_fn(y_pred, y)
                # current_loss.backward()
                # optimizer.step()

                # for v2

                if (epoch + 1) % params["val_patches_saving_epoch"] == 0:
                    saving_flag = True
                else:
                    saving_flag = False

                y_pred, current_loss = mynet(x, d, y, saving_flag)
                current_loss.backward()
                optimizer.step()


                # y_pred_np_saving = other_tools.from_torch_tensor_stack_to_full_res_numpy(y_pred)
                # y_pred_np_saving = y_pred_np_saving[0]
                # exr.write("./train_out_color.exr", y_pred_np_saving[:, :, 0:3])

                # 하나의 배치가 끝날 때 마다의 current loss를 보여줌
                tnr.set_postfix(epoch=epoch, loss=current_loss.item())

                one_epoch_loss += current_loss.data.item()
                num_iter_for_one_epoch += 1

            one_epoch_loss /= num_iter_for_one_epoch
            writer.add_scalar('training loss', one_epoch_loss, epoch)

            "PARAMETER SAVING"
            if (epoch + 1) % params['para_saving_epoch'] == 0:
                out_para_folder_name = params["time_saving_folder"] + "/parameters"
                if not os.path.exists(out_para_folder_name):
                    os.mkdir(out_para_folder_name)
                torch.save(mynet.state_dict(), out_para_folder_name + "/latest_parameter")

            "INTERMEDIATE RESULTING PATCH SAVING"
            if (epoch + 1) % params["val_patches_saving_epoch"] == 0:
                inter_patch_folder_name = params["time_saving_folder"] + "./val_patches"
                if not os.path.exists(inter_patch_folder_name):
                    os.mkdir(inter_patch_folder_name)

                x_np_saving = other_tools.from_torch_tensor_stack_to_full_res_numpy(x)
                d_np_saving = other_tools.from_torch_tensor_stack_to_full_res_numpy(d)
                y_np_saving = other_tools.from_torch_tensor_stack_to_full_res_numpy(y)
                y_pred_np_saving = other_tools.from_torch_tensor_stack_to_full_res_numpy(y_pred)

                for l in range(x_np_saving.shape[0]):
                    exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_color_in.exr",
                              x_np_saving[l, :, :, 0:3])

                    exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_albedo.exr",
                              d_np_saving[l, :, :, 1:4])
                    exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_depth.exr",
                              d_np_saving[l, :, :, 4])
                    exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_g_normal.exr",
                              d_np_saving[l, :, :, 5:8])

                    exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_color_out.exr",
                              y_pred_np_saving[l, :, :, 0:3])

                    exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_color_ref.exr",
                              y_np_saving[l, :, :, 0:3])


    """VALIDATE NETWORK"""
    with torch.no_grad():
        mynet.eval()
        out_folder_name = params["time_saving_folder"] + "/test_imgs"
        if not os.path.exists(out_folder_name):
            os.mkdir(out_folder_name)

        rmse_saving_pth = out_folder_name + "/rmse_list.txt"
        f = open(rmse_saving_pth, 'w')

        image_index = 0

        for data in test_loader:
            x = data['input'].cuda()
            d = data['design'].cuda()
            y = data['target'].cuda()

            # for v1
            # y_pred = mynet(x, d)

            # for v2
            y_pred, current_loss = mynet(x, d, y, True)

            # d_np_saving = other_tools.from_torch_tensor_stack_to_full_res_numpy(d)
            # d_np_saving = d_np_saving[0]
            # exr.write("./test_design_albedo.exr", d_np_saving[:, :, 1:4])
            # exr.write("./test_design_depth.exr", d_np_saving[:, :, 4])
            # exr.write("./test_design_normal.exr", d_np_saving[:, :, 5:8])

            "FROM TORCH TENSOR TO NUMPY TENSOR"
            x_np_saving = other_tools.from_torch_tensor_stack_to_full_res_numpy(x[:, :, :3, :, :])
            y_np_saving = other_tools.from_torch_tensor_stack_to_full_res_numpy(y)
            y_pred_np_saving = other_tools.from_torch_tensor_stack_to_full_res_numpy(y_pred)

            x_np_saving = x_np_saving[0]
            y_np_saving = y_np_saving[0]
            y_pred_np_saving = y_pred_np_saving[0]

            x_np_saving = norm.denormalization_signed_log(x_np_saving)
            y_np_saving = norm.denormalization_signed_log(y_np_saving)
            y_pred_np_saving = norm.denormalization_signed_log(y_pred_np_saving)

            rmse = other_tools.calcRelMSE(y_pred_np_saving, y_np_saving)
            rmse_str = str(image_index) + " image relMSE : " + str(rmse)
            f.write(rmse_str)
            f.write("\n")
            print(rmse_str)

            "SAVING THE RESULTING IMAGES"
            exr.write(out_folder_name + "/" + params['saving_file_name'] + "_" + str(image_index) + "_input.exr",
                      x_np_saving)
            exr.write(out_folder_name + "/" + params['saving_file_name'] + "_" + str(image_index) + "_gt.exr",
                      y_np_saving)
            exr.write(out_folder_name + "/" + params['saving_file_name'] + "_" + str(image_index) + "_result.exr",
                      y_pred_np_saving)

            image_index += 1

        f.close()
    writer.close()
    a=1



def train_test_model_img_v1(train_input_img, train_GT_img, test_input_img, test_GT_img, params):
    """
    입력 구성: path reusing recon에 쓰이는 TRAIN AND TEST 버퍼.
    순서: normalization -> making gird by order -> design matrix -> data loader -> network setting -> train -> test
    특징: 함수의 이름에서 알 수 있듯이 stack으로 받는 것이 아닌 img형태로 데이터를 받음
    """

    """INITIAL SETTING"""
    # GPU index setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_train, H, W, _ = train_input_img.shape
    N_test = test_input_img.shape[0]
    H_d = int(H // params["tile_length"])
    W_d = int(W // params["tile_length"])


    """NORMALIZATION FOR STACK BUFFER"""
    norm.normalize_input_img_v1(train_input_img)
    norm.normalize_input_img_v1(test_input_img)

    norm.normalize_GT_v1(train_GT_img)
    norm.normalize_GT_v1(test_GT_img)


    """MAKING THE DESIGN MATRIX"""
    train_design_stack = design.generate_design_mat_from_img_v1(train_input_img[:, :, :, 3:],
                                                       params['tile_length'], params['grid_order'], False)
    test_design_stack = design.generate_design_mat_from_img_v1(test_input_img[:, :, :, 3:],
                                                      params['tile_length'], params['grid_order'], False)


    """SETTING DATA LOAD AND CORRESPONDING TRANSFORMS"""
    # define transform op
    transform_patch = transforms.Compose([
        FT.RandomCrop_img_stack_with_design(params['patch_size'], params['tile_length']),
        FT.RandomFlip_with_design(multi_crop=False),
        FT.ToTensor_img_stack_with_design(multi_crop=False)
    ])
    transform_img = transforms.Compose([FT.ToTensor_img_stack_with_design(multi_crop=False)])  # targeting for image

    # train data loader
    train_data = dataset.Supervised_dataset_with_design_v1(train_input_img, train_design_stack, train_GT_img,
                                               train=True, transform=transform_patch)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)

    # test data loader
    test_data = dataset.Supervised_dataset_with_design_v1(test_input_img, test_design_stack, test_GT_img,
                                                           train=False, transform=transform_img)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)


    """NETWORK INITIALIZATION"""
    mynet = models_v1.NPR_net_img_v1(channels_in=10, kernel_size=3, tile_length=4, n_layers=20, length_inter_tile=5,
                                     epsilon=0.01, pad_mode=1).train().to(device)

    """SAVING THE TENSORBOARD"""
    out_tensorboard_folder_name = "./results/tensorboards/" + params['saving_folder_name']
    if not os.path.exists(out_tensorboard_folder_name):
        os.mkdir(out_tensorboard_folder_name)
    writer = tensorboard.SummaryWriter(out_tensorboard_folder_name + "/" + params['saving_file_name'] + "_" +
                                       str(datetime.today().strftime("%Y_%m_%d_%H_%M")))


    """SET LOSS AND OPTIMIZATION"""
    loss_fn = my_loss.loss_for_stit_v1(params['tile_length'], params["stitching_weights"], params['loss_type'])
    optimizer = optim.Adam(mynet.parameters(), lr=params['lr'])


    """TRAIN NETWORK"""
    epochs = params["epochs"]

    with tqdm(range(0, epochs), leave=True) as tnr:
        tnr.set_postfix(epoch=0, loss=-1.)

        for epoch in tnr:

            one_epoch_loss = 0.0
            num_iter_for_one_epoch = 0

            for data in train_loader:
                optimizer.zero_grad()

                x = data['input'].cuda()
                d = data['design'].cuda()
                y = data['target'].cuda()

                y_pred = mynet(x, d)

                current_loss = loss_fn(y_pred, y)
                current_loss.backward()
                optimizer.step()

                # 하나의 배치가 끝날 때 마다의 current loss를 보여줌
                tnr.set_postfix(epoch=epoch, loss=current_loss.item())

                one_epoch_loss += current_loss.data.item()
                num_iter_for_one_epoch += 1

            one_epoch_loss /= num_iter_for_one_epoch
            writer.add_scalar('training loss', one_epoch_loss, epoch)

            "PARAMETER SAVING"
            if (epoch + 1) % params['para_saving_epoch'] == 0:
                out_para_folder_name = "./results/parameters/" + params['saving_folder_name']
                if not os.path.exists(out_para_folder_name):
                    os.mkdir(out_para_folder_name)
                torch.save(mynet.state_dict(), out_para_folder_name + "/latest_parameter")


    """VALIDATE NETWORK"""
    with torch.no_grad():
        mynet.eval()

        out_folder_name = "./results/imgs/" + params['saving_folder_name']
        if not os.path.exists(out_folder_name):
            os.mkdir(out_folder_name)

        time_folder_name = out_folder_name + "/" + str(datetime.today().strftime("%Y_%m_%d_%H_%M"))
        if not os.path.exists(time_folder_name):
            os.mkdir(time_folder_name)

        rmse_saving_pth = time_folder_name + "/rmse_list.txt"
        f = open(rmse_saving_pth, 'w')

        image_index = 0

        for data in test_loader:
            x = data['input'].cuda()
            d = data['design'].cuda()
            y = data['target'].cuda()

            y_pred = mynet(x, d)

            # d_np_saving = other_tools.from_torch_tensor_stack_to_full_res_numpy(d)
            # d_np_saving = d_np_saving[0]
            # exr.write("./test_design_albedo.exr", d_np_saving[:, :, 1:4])
            # exr.write("./test_design_depth.exr", d_np_saving[:, :, 4])
            # exr.write("./test_design_normal.exr", d_np_saving[:, :, 5:8])

            "FROM TORCH TENSOR TO NUMPY TENSOR"
            x_np_saving = other_tools.from_torch_tensor_stack_to_full_res_numpy(x[:, :, :3, :, :])
            y_np_saving = other_tools.from_torch_tensor_stack_to_full_res_numpy(y)
            y_pred_np_saving = other_tools.from_torch_tensor_stack_to_full_res_numpy(y_pred)

            x_np_saving = x_np_saving[0]
            y_np_saving = y_np_saving[0]
            y_pred_np_saving = y_pred_np_saving[0]

            x_np_saving = norm.denormalization_signed_log(x_np_saving)
            y_np_saving = norm.denormalization_signed_log(y_np_saving)
            y_pred_np_saving = norm.denormalization_signed_log(y_pred_np_saving)

            rmse = other_tools.calcRelMSE(y_pred_np_saving, y_np_saving)
            rmse_str = str(image_index) + " image relMSE : " + str(rmse)
            f.write(rmse_str)
            f.write("\n")
            print(rmse_str)

            "SAVING THE RESULTING IMAGES"
            exr.write(time_folder_name + "/" + params['saving_file_name'] + "_" + str(image_index) + "_input.exr",
                      x_np_saving)
            exr.write(time_folder_name + "/" + params['saving_file_name'] + "_" + str(image_index) + "_gt.exr",
                      y_np_saving)
            exr.write(time_folder_name + "/" + params['saving_file_name'] + "_" + str(image_index) + "_result.exr",
                      y_pred_np_saving)

            image_index += 1

        f.close()
    writer.close()
    a=1



def test_model_stack_v1(test_input_stack, test_design_stack, test_GT_stack, params):
    """
    입력 구성: 오직 TEST 버퍼.
    순서: normalization -> making gird by order -> design matrix -> data loader -> network setting -> train -> test
    특징: param에 있는 trained model path에 따라 얻어진 모델 테스트
    """

    """INITIAL SETTING"""
    # GPU index setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_test, H_d, W_d, tile_size, _ = test_input_stack.shape
    H = H_d * params["tile_length"]
    W = W_d * params["tile_length"]


    """SETTING DATA LOAD AND CORRESPONDING TRANSFORMS"""
    # define transform op
    transform_img = transforms.Compose([FT.ToTensor_stack_with_design(multi_crop=False)])  # targeting for image

    # test data loader
    test_data = dataset.Supervised_dataset_with_design_v1(test_input_stack, test_design_stack, test_GT_stack,
                                                           train=False, transform=transform_img)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)


    """NETWORK INITIALIZATION"""
    mynet = models_v1.NPR_net_stack_v1(ch_in=10, kernel_size=3, tile_length=4, n_layers=12, length_inter_tile=5,
                                       epsilon=0.01, pad_mode=1, no_stit_input=params['no_boundary_for_input'],
                                       no_stit_design=params['no_boundary_for_design']).train().to(device)

    """LOAD THE TRAINED PARAMETER"""
    parameter_pth = "./results/parameters/" + params['trained_parameter_pth'] + "/" + params['trained_parameter_name']
    mynet.load_state_dict(torch.load(parameter_pth))

    """TEST NETWORK"""
    with torch.no_grad():
        mynet.eval()

        out_folder_name = "./results/imgs/" + params['saving_folder_name']
        if not os.path.exists(out_folder_name):
            os.mkdir(out_folder_name)

        time_folder_name = out_folder_name + "/" + str(datetime.today().strftime("%Y_%m_%d_%H_%M"))
        if not os.path.exists(time_folder_name):
            os.mkdir(time_folder_name)

        rmse_saving_pth = time_folder_name + "/rmse_list.txt"
        f = open(rmse_saving_pth, 'w')

        image_index = 0

        for data in test_loader:
            x = data['input'].cuda()
            d = data['design'].cuda()
            y = data['target'].cuda()

            y_pred = mynet(x, d)

            "FROM TORCH TENSOR TO NUMPY TENSOR"
            x_np_saving = other_tools.from_torch_tensor_stack_to_full_res_numpy(x[:, :, :3, :, :])
            y_np_saving = other_tools.from_torch_tensor_stack_to_full_res_numpy(y)
            y_pred_np_saving = other_tools.from_torch_tensor_stack_to_full_res_numpy(y_pred)

            x_np_saving = x_np_saving[0]
            y_np_saving = y_np_saving[0]
            y_pred_np_saving = y_pred_np_saving[0]

            x_np_saving = norm.denormalization_signed_log(x_np_saving)
            y_np_saving = norm.denormalization_signed_log(y_np_saving)
            y_pred_np_saving = norm.denormalization_signed_log(y_pred_np_saving)

            rmse = other_tools.calcRelMSE(y_pred_np_saving, y_np_saving)
            rmse_str = str(image_index) + " image relMSE : " + str(rmse)
            f.write(rmse_str)
            f.write("\n")
            print(rmse_str)

            "SAVING THE RESULTING IMAGES"
            exr.write(time_folder_name + "/" + params['saving_file_name'] + "_" + str(image_index) + "_input.exr",
                      x_np_saving)
            exr.write(time_folder_name + "/" + params['saving_file_name'] + "_" + str(image_index) + "_gt.exr",
                      y_np_saving)
            exr.write(time_folder_name + "/" + params['saving_file_name'] + "_" + str(image_index) + "_result.exr",
                      y_pred_np_saving)

            image_index += 1

        f.close()
