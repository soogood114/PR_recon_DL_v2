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
import Feed_and_Loss.feed_transform as FT
import Feed_and_Loss.dataset as dataset
import Feed_and_Loss.loss as my_loss

import Models.NGPT_models as NGPT


def train_test_cmp_model_img_v1(train_input_img_buffer, train_ref_img_buffer, test_input_img_buffer,
                                test_ref_img_buffer, params):
    """
    입력 구성: NGPT에 쓰이는 TRAIN AND TEST 버퍼.
    특징 #1 : 기존의 함수의 형태는 유지를 하고 NGPT에 맞게 FULL IMG로 학습이 진행이 됨.
    """

    """INITIAL SETTING"""
    # GPU index setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_train, H, W, ch_in = train_input_img_buffer.shape
    N_test = test_input_img_buffer.shape[0]

    """NORMALIZATION AND BUFFER SELECTION"""
    norm.normalize_input_img_cmp(train_input_img_buffer, train_ref_img_buffer)
    norm.normalize_input_img_cmp(test_input_img_buffer, test_ref_img_buffer)

    """SETTING DATA LOAD AND CORRESPONDING TRANSFORMS"""
    # define transform op
    transform_patch = transforms.Compose([
        FT.RandomCrop(params['patch_size']),
        # FT.RandomFlip_with_design(multi_crop=False),  # 현재 문제가 있음.
        FT.ToTensor(multi_crop=False)
    ])
    transform_img = transforms.Compose([FT.ToTensor(multi_crop=False)])  # targeting for image

    # train data loader
    train_data = dataset.Supervised_dataset(train_input_img_buffer, train_ref_img_buffer
                                            , train=True, transform=transform_patch)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)

    # test data loader
    test_data = dataset.Supervised_dataset(test_input_img_buffer, test_ref_img_buffer,
                                           train=False, transform=transform_img)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    """NETWORK INITIALIZATION"""
    mynet = NGPT.Back_bone_NGPT_v1(params, channels_in=ch_in, out_dim=3).train().to(device)

    """SAVING THE TENSORBOARD"""
    out_tensorboard_folder_name = params["time_saving_folder"] + "/tensorboards"
    if not os.path.exists(out_tensorboard_folder_name):
        os.mkdir(out_tensorboard_folder_name)
    writer = tensorboard.SummaryWriter(out_tensorboard_folder_name)

    """SET LOSS AND OPTIMIZATION"""
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
                y = data['target'].cuda()

                y_pred = mynet(x)
                current_loss = mynet.loss(y_pred, y)
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
                out_para_folder_name = params["time_saving_folder"] + "/parameters"
                if not os.path.exists(out_para_folder_name):
                    os.mkdir(out_para_folder_name)
                torch.save(mynet.state_dict(), out_para_folder_name + "/latest_parameter")

            "INTERMEDIATE RESULTING PATCH SAVING"
            if (epoch + 1) % params["val_patches_saving_epoch"] == 0:
                inter_patch_folder_name = params["time_saving_folder"] + "./val_patches"
                if not os.path.exists(inter_patch_folder_name):
                    os.mkdir(inter_patch_folder_name)

                x_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(x)
                y_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(y)
                y_pred_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(y_pred)

                for l in range(x_np_saving.shape[0]):
                    exr.write(inter_patch_folder_name + "/epoch_" + str(epoch) + "_" + str(l) + "_color_in.exr",
                              x_np_saving[l, :, :, 0:3])

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
            y = data['target'].cuda()

            y_pred = mynet(x)

            "FROM TORCH TENSOR TO NUMPY TENSOR"
            x_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(x[:, :3, :, :])
            y_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(y)
            y_pred_np_saving = other_tools.from_torch_tensor_img_to_full_res_numpy(y_pred)

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
