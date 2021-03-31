import torch.nn as nn
import torch.nn.functional as F
import torch
import Feed_and_Loss.loss as my_loss

class NGPT_PU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NGPT_PU, self).__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_dim, 2 * out_dim, 1),
            nn.LeakyReLU()
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(2 * out_dim, out_dim, 3, padding=1),
            nn.LeakyReLU()
        )


    def forward(self, x):
        # y = self.conv1x1(x)
        # y = self.conv3x3(y)
        return self.conv3x3(self.conv1x1(x))


class Back_bone_NGPT_v1(nn.Module):
    """네트워크 설명"""
    """ 인풋 : 각각 들어간다.  아웃풋 : 타일"""
    """ SIGA19에서 나온 supervised 논문에서의 네트워크를 구현"""

    def __init__(self, params, channels_in=5, out_dim=3):
        super(Back_bone_NGPT_v1, self).__init__()

        self.channels_in = channels_in  # RGB + Features

        # self.Kernel_size = kernel_size
        # self.k_size = pred_kernel

        """First encoding block"""
        self.en1_PU1 = NGPT_PU(channels_in, 40)
        self.en1_PU2 = NGPT_PU(channels_in + 40, 40)
        self.en1_PU3 = NGPT_PU(channels_in + 40 * 2, 40)
        self.en1_PU4 = NGPT_PU(channels_in + 40 * 3, 40)

        # down
        self.en1_down_conv = nn.Conv2d(40 * 4 + channels_in, 160, 2, stride=2)
        self.en1_down_conv_relu = nn.LeakyReLU()
        self.en1_num_ch = 40 * 4 + channels_in


        """Second encoding block"""
        self.en2_PU1 = NGPT_PU(160, 80)
        self.en2_PU2 = NGPT_PU(160 + 80, 80)
        self.en2_PU3 = NGPT_PU(160 + 80 * 2, 80)

        # down
        self.en2_down_conv = nn.Conv2d(80 * 3 + 160, 160, 2, stride=2)
        self.en2_down_conv_relu = nn.LeakyReLU()
        self.en2_num_ch = 80 * 3 + 160


        """Third encoding block"""
        self.en3_PU1 = NGPT_PU(160, 80)
        self.en3_PU2 = NGPT_PU(160 + 80, 80)

        # down
        self.en3_down_conv = nn.Conv2d(80 * 2 + 160, 160, 2, stride=2)
        self.en3_down_conv_relu = nn.LeakyReLU()
        self.en3_num_ch = 80 * 2 + 160


        """Latent encoding block"""
        self.l_PU1 = NGPT_PU(160, 80)
        self.l_PU2 = NGPT_PU(160 + 80, 80)
        self.l_PU3 = NGPT_PU(160 + 80 * 2, 80)
        self.l_PU4 = NGPT_PU(160 + 80 * 3, 80)

        # up
        self.l_up_conv = nn.ConvTranspose2d(80 * 4 + 160, 160, 2, stride=2, padding=0)
        self.l_up_conv_relu = nn.LeakyReLU()
        self.l_num_ch = 80 * 4 + 160


        """First decoding block"""
        self.de1_PU1 = NGPT_PU(160 + self.en3_num_ch, 80)
        self.de1_PU2 = NGPT_PU(160 + self.en3_num_ch + 80, 80)

        # up
        self.de1_up_conv = nn.ConvTranspose2d(160 + self.en3_num_ch + 80 * 2, 160, 2, stride=2, padding=0)
        self.de1_up_conv_relu = nn.LeakyReLU()


        """Second decoding block"""
        self.de2_PU1 = NGPT_PU(160 + self.en2_num_ch, 80)
        self.de2_PU2 = NGPT_PU(160 + self.en2_num_ch + 80, 80)
        self.de2_PU3 = NGPT_PU(160 + self.en2_num_ch + 80 * 2, 80)

        # up
        self.de2_up_conv = nn.ConvTranspose2d(160 + self.en2_num_ch + 80 * 3, 80, 2, stride=2, padding=0)
        self.de2_up_conv_relu = nn.LeakyReLU()

        """Third decoding block"""
        self.de3_PU1 = NGPT_PU(80 + self.en1_num_ch, 40)
        self.de3_PU2 = NGPT_PU(80 + self.en1_num_ch + 40, 40)
        self.de3_PU3 = NGPT_PU(80 + self.en1_num_ch + 40 * 2, 40)
        self.de3_PU4 = NGPT_PU(80 + self.en1_num_ch + 40 * 3, 40)

        # down -> out_ch
        self.de3_up_conv = nn.Conv2d(80 + self.en1_num_ch + 40 * 4, out_dim, 3, padding=1)
        self.de3_up_conv_relu = nn.ReLU()

        "NEW ONE"


    def loss(self, img_out, ref):
        # l1
        loss = torch.mean(torch.abs(img_out - ref))

        # SMAPE loss
        # loss = torch.mean(torch.div(torch.abs(img_out - ref), (torch.abs(img_out) + torch.abs(ref) + 0.01)))

        return loss


    def get_loss(self, out, ref):
        "이건 PR net 비교를 위해 새로 추가 됨"

        ref_col = ref[:, :3, :, :]

        data_loss = self.loss(out, ref_col)

        return data_loss

    def forward(self, input):

        """First encoding block"""
        o_en1 = torch.cat((input, self.en1_PU1(input)), dim=1)

        o_en1 = torch.cat((o_en1, self.en1_PU2(o_en1)), dim=1)

        o_en1 = torch.cat((o_en1, self.en1_PU3(o_en1)), dim=1)

        o_en1 = torch.cat((o_en1, self.en1_PU4(o_en1)), dim=1)

        o_en1_D = self.en1_down_conv_relu(self.en1_down_conv(o_en1))

        """Second encoding block"""
        o_en2 = torch.cat((o_en1_D, self.en2_PU1(o_en1_D)), dim=1)

        o_en2 = torch.cat((o_en2, self.en2_PU2(o_en2)), dim=1)

        o_en2 = torch.cat((o_en2, self.en2_PU3(o_en2)), dim=1)

        o_en2_D = self.en2_down_conv_relu(self.en2_down_conv(o_en2))

        """Third encoding block"""
        o_en3 = torch.cat((o_en2_D, self.en3_PU1(o_en2_D)), dim=1)

        o_en3 = torch.cat((o_en3, self.en3_PU2(o_en3)), dim=1)

        o_en3_D = self.en3_down_conv_relu(self.en3_down_conv(o_en3))


        """Latent encoding block"""
        o_l = torch.cat((o_en3_D, self.l_PU1(o_en3_D)), dim=1)

        o_l = torch.cat((o_l, self.l_PU2(o_l)), dim=1)

        o_l = torch.cat((o_l, self.l_PU3(o_l)), dim=1)

        o_l = torch.cat((o_l, self.l_PU4(o_l)), dim=1)

        # o_l_U = self.l_up_conv_relu(self.l_up_conv(o_l))


        """First decoding block"""
        o_de1 = torch.cat((o_en3, self.l_up_conv_relu(self.l_up_conv(o_l))), dim=1)

        o_de1 = torch.cat((o_de1,  self.de1_PU1(o_de1)), dim=1)

        o_de1 = torch.cat((o_de1, self.de1_PU2(o_de1)), dim=1)

        # o_de1_U = self.de1_up_conv_relu(self.de1_up_conv(o_de1))


        """Second decoding block"""
        o_de2 = torch.cat((o_en2, self.de1_up_conv_relu(self.de1_up_conv(o_de1))), dim=1)

        o_de2 = torch.cat((o_de2, self.de2_PU1(o_de2)), dim=1)

        o_de2 = torch.cat((o_de2, self.de2_PU2(o_de2)), dim=1)

        o_de2 = torch.cat((o_de2, self.de2_PU3(o_de2)), dim=1)

        # o_de2_U = self.de2_up_conv_relu(self.de2_up_conv(o_de2))


        """Third decoding block"""
        o_de3 = torch.cat((o_en1, self.de2_up_conv_relu(self.de2_up_conv(o_de2))), dim=1)

        o_de3 = torch.cat((o_de3, self.de3_PU1(o_de3)), dim=1)

        o_de3 = torch.cat((o_de3, self.de3_PU2(o_de3)), dim=1)

        o_de3 = torch.cat((o_de3, self.de3_PU3(o_de3)), dim=1)

        o_de3 = torch.cat((o_de3, self.de3_PU4(o_de3)), dim=1)
        final_out = self.de3_up_conv_relu(self.de3_up_conv(o_de3))

        return final_out


class Back_bone_NGPT_v2(nn.Module):
    """네트워크 설명"""
    """ 인풋 : 각각 들어간다.  아웃풋 : 타일"""
    """ SIGA19에서 나온 supervised 논문에서의 네트워크를 구현"""
    """ v1과는 다른 것은 layer가 적어서 좀더 가볍다는 것."""

    def __init__(self, channels_in=5, out_dim=3):
        super(Back_bone_NGPT_v2, self).__init__()

        self.channels_in = channels_in  # RGB + Features

        # self.Kernel_size = kernel_size
        # self.k_size = pred_kernel

        """First encoding block"""
        self.en1_PU1 = NGPT_PU(channels_in, 20)
        self.en1_PU2 = NGPT_PU(channels_in + 20, 20)
        self.en1_PU3 = NGPT_PU(channels_in + 20 * 2, 20)
        self.en1_PU4 = NGPT_PU(channels_in + 20 * 3, 20)

        # down
        self.en1_down_conv = nn.Conv2d(20 * 4 + channels_in, 80, 2, stride=2)
        self.en1_down_conv_relu = nn.LeakyReLU()
        self.en1_num_ch = 20 * 4 + channels_in

        """Second encoding block"""
        self.en2_PU1 = NGPT_PU(80, 40)
        self.en2_PU2 = NGPT_PU(80 + 40, 40)
        self.en2_PU3 = NGPT_PU(80 + 40 * 2, 40)

        # down
        self.en2_down_conv = nn.Conv2d(40 * 3 + 80, 80, 2, stride=2)
        self.en2_down_conv_relu = nn.LeakyReLU()
        self.en2_num_ch = 40 * 3 + 80

        """Latent encoding block"""
        self.l_PU1 = NGPT_PU(80, 40)
        self.l_PU2 = NGPT_PU(80 + 40, 40)
        self.l_PU3 = NGPT_PU(80 + 40 * 2, 40)
        self.l_PU4 = NGPT_PU(80 + 40 * 3, 40)

        # up
        self.l_up_conv = nn.ConvTranspose2d(40 * 4 + 80, 80, 2, stride=2, padding=0)
        self.l_up_conv_relu = nn.LeakyReLU()
        self.l_num_ch = 40 * 4 + 80

        """Second decoding block"""
        self.de2_PU1 = NGPT_PU(80 + self.en2_num_ch, 40)
        self.de2_PU2 = NGPT_PU(80 + self.en2_num_ch + 40, 40)
        self.de2_PU3 = NGPT_PU(80 + self.en2_num_ch + 40 * 2, 40)

        # up
        self.de2_up_conv = nn.ConvTranspose2d(80 + self.en2_num_ch + 40 * 3, 40, 2, stride=2, padding=0)
        self.de2_up_conv_relu = nn.LeakyReLU()

        """Third decoding block"""
        self.de3_PU1 = NGPT_PU(40 + self.en1_num_ch, 20)
        self.de3_PU2 = NGPT_PU(40 + self.en1_num_ch + 20, 20)
        self.de3_PU3 = NGPT_PU(40 + self.en1_num_ch + 20 * 2, 20)
        self.de3_PU4 = NGPT_PU(40 + self.en1_num_ch + 40 * 3, 20)

        # down -> out_ch
        self.de3_up_conv = nn.Conv2d(40 + self.en1_num_ch + 20 * 4, out_dim, 3, padding=1)
        self.de3_up_conv_relu = nn.LeakyReLU()

    def forward(self, input):
        """First encoding block"""
        o_en1 = torch.cat((input, self.en1_PU1(input)), dim=1)

        o_en1 = torch.cat((o_en1, self.en1_PU2(o_en1)), dim=1)

        o_en1 = torch.cat((o_en1, self.en1_PU3(o_en1)), dim=1)

        o_en1 = torch.cat((o_en1, self.en1_PU4(o_en1)), dim=1)

        o_en1_D = self.en1_down_conv_relu(self.en1_down_conv(o_en1))

        """Second encoding block"""
        o_en2 = torch.cat((o_en1_D, self.en2_PU1(o_en1_D)), dim=1)

        o_en2 = torch.cat((o_en2, self.en2_PU2(o_en2)), dim=1)

        o_en2 = torch.cat((o_en2, self.en2_PU3(o_en2)), dim=1)

        o_en2_D = self.en2_down_conv_relu(self.en2_down_conv(o_en2))

        """Latent encoding block"""
        o_l = torch.cat((o_en2_D, self.l_PU1(o_en2_D)), dim=1)

        o_l = torch.cat((o_l, self.l_PU2(o_l)), dim=1)

        o_l = torch.cat((o_l, self.l_PU3(o_l)), dim=1)

        o_l = torch.cat((o_l, self.l_PU4(o_l)), dim=1)

        o_l_U = self.l_up_conv_relu(self.l_up_conv(o_l))

        """Second decoding block"""
        o_de2 = torch.cat((o_en2, o_l_U), dim=1)

        o_de2 = torch.cat((o_de2, self.de2_PU1(o_de2)), dim=1)

        o_de2 = torch.cat((o_de2, self.de2_PU2(o_de2)), dim=1)

        o_de2 = torch.cat((o_de2, self.de2_PU3(o_de2)), dim=1)

        o_de2_U = self.de2_up_conv_relu(self.de2_up_conv(o_de2))

        """Third decoding block"""
        o_de3 = torch.cat((o_en1, o_de2_U), dim=1)

        o_de3 = torch.cat((o_de3, self.de3_PU1(o_de3)), dim=1)

        o_de3 = torch.cat((o_de3, self.de3_PU2(o_de3)), dim=1)

        o_de3 = torch.cat((o_de3, self.de3_PU3(o_de3)), dim=1)

        o_de3 = torch.cat((o_de3, self.de3_PU4(o_de3)), dim=1)
        final_out = self.de3_up_conv_relu(self.de3_up_conv(o_de3))

        return final_out
