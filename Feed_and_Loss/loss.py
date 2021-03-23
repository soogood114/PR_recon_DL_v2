import torch
import torch.nn.functional as F
import torch.utils.data


def CalcGrad_torch_tensor(data):
    s, c, h, w = data.shape
    dX = data[:, :, :, 1:] - data[:, :, :, :w - 1]
    dY = data[:, :, 1:, :] - data[:, :, :h - 1, :]

    dX = torch.cat((dX, torch.zeros(s, c, h, 1).cuda()), dim=3)
    dY = torch.cat((dY, torch.zeros(s, c, 1, w).cuda()), dim=2)

    return dX, dY


class my_custom_loss_v1(object):
    """
    Args:
        num_tiels_HW : 타일 단위로 쪼갤 것인데 H방향으로 몇개의 타일, W 방향으로 몇개의 타일 이렇게 정의가 된다.
    """

    def __init__(self, user_weights, SMAP_on, reinhard_loss):
        self.colorLoss_weight = user_weights[0]
        self.gradLoss_weight = user_weights[1]

        self.SMAPE_on = SMAP_on
        self.reinhard_loss = reinhard_loss

    def SMAPE_loss(self, output, target):
        loss = torch.mean(torch.div(torch.abs(output - target), (torch.abs(output) + torch.abs(target) + 0.01)))
        return loss

    def ReinhardTransform(self, output, target):
        """
        Args:
            ReinhardTransform를 수행해 loss를 구하기 전에 loss의 입력을 안정화 시킴
            이는 sig19 ngpt에서 나옴
        """
        return torch.div(output, (1.0 + torch.mean(torch.abs(output)))), \
               torch.div(target, (1.0 + torch.mean(torch.abs(target))))



    def __call__(self, output, target):
        output_dx, output_dy = CalcGrad_torch_tensor(output)
        target_dx, target_dy = CalcGrad_torch_tensor(target)

        W_l1_color = self.colorLoss_weight
        W_l1_grad = self.gradLoss_weight

        if self.reinhard_loss == True:
            output, target = self.ReinhardTransform(output, target)
            output_dx, target_dx = self.ReinhardTransform(output_dx, target_dx)
            output_dy, target_dy = self.ReinhardTransform(output_dy, target_dy)

        if self.SMAPE_on == True:
            color_loss = self.SMAPE_loss(output, target)
        else:
            color_loss = torch.mean(torch.abs(output - target))

        L1_dx = torch.mean(torch.abs(output_dx - target_dx))
        L1_dy = torch.mean(torch.abs(output_dy - target_dy))

        return W_l1_color * color_loss + W_l1_grad * (L1_dx + L1_dy)



class loss_for_stit_v1(object):
    """
    input : stack version of output and target
    output : various losses with stitching
    feature #1 : input과 target 모두 stack version이어야 함.
    feature #2 : 필요에 따라 stitching   을 고려할 수 있음.
    """

    def __init__(self, tile_length=4, weight_for_stitching=0, loss_type="l1"):
        self.weight_for_stitching = weight_for_stitching
        self.loss_type = loss_type
        self.tile_length = tile_length
        self.tile_size = tile_length ** 2
        self.tile_size_stit = tile_length ** 2 + tile_length * 2

    def __call__(self, output, target):

        if self.weight_for_stitching == 0:
            loss = self.Get_loss(output, target)
        else:
            data_loss = self.Get_loss(output, target)
            stitching_loss = self.Get_stitching_loss(output)
            loss = data_loss + stitching_loss * self.weight_for_stitching

        return loss

    def Get_stitching_loss(self, output):
        """
        초기 버전처럼 stitching으로 서로 겹치는 부분은 같게 함.
        stit : 바운더리 부분
        ori : 원래 타일 안 부분
        """

        s = self.tile_length
        t = self.tile_size

        b, _, c, h_d, w_d = output.size()

        stit_out_x = output[:, t:t + s, :, :, :]
        stit_out_y = output[:, t + s:, :, :, :]

        ori_out = output[:, :t, :, :, :].view(b ,s, s, c, h_d, w_d)
        ori_out_x = ori_out[:, :, 0, :, :, :]
        ori_out_y = ori_out[:, :, :, 0, :, :]

        if self.loss_type == "l1":
            stitching_loss = self.l1_loss(stit_out_x[:, :, :, :, :-1], ori_out_x[:, :, :, :, 1:]) +\
                             self.l1_loss(stit_out_y[:, :, :, :-1, :], ori_out_y[:, :, :, 1:, :])
        elif self.loss_type == "l2":
            stitching_loss = self.l2_loss(stit_out_x[:, :, :, :, :-1], ori_out_x[:, :, :, :, 1:]) + \
                             self.l2_loss(stit_out_y[:, :, :, :-1, :], ori_out_y[:, :, :, 1:, :])
        elif self.loss_type == "smape":
            stitching_loss = self.SMAPE_loss(stit_out_x[:, :, :, :, :-1], ori_out_x[:, :, :, :, 1:]) + \
                             self.SMAPE_loss(stit_out_y[:, :, :, :-1, :], ori_out_y[:, :, :, 1:, :])
        else:
            stitching_loss = self.l1_loss(stit_out_x[:, :, :, :, :-1], ori_out_x[:, :, :, :, 1:]) + \
                             self.l1_loss(stit_out_y[:, :, :, :-1, :], ori_out_y[:, :, :, 1:, :])

        return stitching_loss


    def Get_loss(self, input, target):
        if self.loss_type == "l1":
            data_loss = self.l1_loss(input, target)
        elif self.loss_type == "l2":
            data_loss = self.l2_loss(input, target)
        elif self.loss_type == "smape":
            data_loss = self.SMAPE_loss(input, target)
        else:
            data_loss = self.l1_loss(input, target)

        return data_loss


    def SMAPE_loss(self, output, target):
        loss = torch.mean(torch.div(torch.abs(output - target), (torch.abs(output) + torch.abs(target) + 0.01)))
        return loss

    def l1_loss(self, output, target):
        return torch.mean(torch.abs(output - target))

    def l2_loss(self, output, target):
        return torch.mean(torch.square(output - target))

