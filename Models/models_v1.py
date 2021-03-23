import torch.nn as nn
import torch.nn.functional as F
import torch

import Feed_and_Loss.loss as my_loss

import Models.net_op as net_op
import Data.exr as exr
import Data.other_tools as other_tools


class NPR_net_stack_v1(nn.Module):
    """
    네트워크 개요
    - 기존 PR_recon_DL 에서 "PR_net_input_tr_tile_by_tile_stit_v2"의 구조를 활용함.
    - 가장 기본이 되는 네트워크로 간단한 기능만 유지.
    - 교수님의 피드백을 받아 no boudnary라는 기능을 추가함.
    """

    def __init__(self, ch_in=10, kernel_size=3, tile_length=4, n_layers=12, length_inter_tile=5, epsilon=0.01,
                 pad_mode=0, no_stit_input=True, no_stit_design=True):
        super(NPR_net_stack_v1, self).__init__()

        self.ch_in = ch_in
        self.k_size = kernel_size
        self.tile_length = tile_length

        self.tile_size = tile_length ** 2
        self.tile_size_stit = tile_length ** 2 + tile_length * 2

        self.epsilon = epsilon

        self.pad_mode = pad_mode  # 0: zero, 1: reflected, 2: circular

        self.length_inter = length_inter_tile
        self.inter_tile_num = int(length_inter_tile ** 2)

        self.no_stit_input = no_stit_input
        self.no_stit_design = no_stit_design


        if no_stit_input:
            self.start_ch = ch_in * self.tile_size
            self.final_ch = int((self.tile_size ** 2) * self.inter_tile_num)

        else:
            self.start_ch = ch_in * self.tile_size_stit
            self.final_ch = int((self.tile_size_stit ** 2) * self.inter_tile_num)


        self.inter_ch = 1024

        self.layers = [nn.Conv2d(self.start_ch, self.inter_ch, kernel_size, padding=(kernel_size - 1) // 2),
                       nn.LeakyReLU()]

        for l in range(n_layers - 2):
            self.layers += [
                nn.Conv2d(self.inter_ch, self.inter_ch, kernel_size, padding=(kernel_size - 1) // 2),
                nn.LeakyReLU()
            ]

        self.feature_layers_feed = nn.Sequential(*self.layers)  # to get the feature

        self.layers_for_weights_feed = nn.Conv2d(self.inter_ch, self.final_ch, kernel_size,
                                            padding=(kernel_size - 1) // 2)

        # self.layers_for_weights_feed.weight.data.fill_(0.0)
        # self.layers_for_weights_feed.bias.data.fill_(0.0)


    def forward(self, input, design):
        """
        input : B T(tile_size) C_in H_d W_d
        design : B T(tile_size) C_de H_d W_d
        """

        "INITIAL SETTING"
        b = input.size(0)
        ch_in = input.size(2)  # 10 (color + g_buffer)
        ch_de = design.size(2)
        h_d, w_d = input.size(3), input.size(4)
        hw_d = h_d * w_d
        t_de = design.size(1)

        s = self.tile_length
        # t = self.tile_size_stit  # 24
        t = input.size(1)

        length_inter = self.length_inter  # length of inter tile
        num_inter = length_inter ** 2

        # domain for prediction using parameters which is the result of least square
        domain = design.permute(0, 3, 4, 1, 2).contiguous().view(b * hw_d, t_de, ch_de)

        # make sure to match input tile_size
        design = design[:, :t, :, :, :].contiguous()


        out = torch.zeros((b, t_de, 3, h_d, w_d), dtype=input.dtype, layout=input.layout,
                          device=input.device)

        "UNFOLD DESIGN MATRIX"
        design = design.view(b, t * ch_de, h_d, w_d)  # 5D -> 4D
        design = self.unfold_and_padding(design)  # b, t * ch_de * num_inter, hw_d

        "X FROM DESIGN MATRIX"
        design = design.permute(0, 2, 1).contiguous().view(b * hw_d, t * ch_de, num_inter)
        design = design.permute(0, 2, 1).contiguous().view(b * hw_d * num_inter, t * ch_de)
        X = design.view(-1, t, ch_de)

        # TEST
        X = self.norm_in_prediction_window(X.view(b * hw_d, num_inter, t, ch_de))
        X = X.view(b * hw_d * num_inter, t, ch_de)
        #


        "GET THE FEATURE FROM FEATURE NETWORK"
        feature = self.feature_layers_feed(input.view(b, t * ch_in, h_d, w_d))

        "W FROM THE FEATURE"  # 여기서 메모리가 뻥튀기 됨.
        # W = torch.tanh(self.layers_for_weights_feed(feature))
        W = self.layers_for_weights_feed(feature)
        W = W.view(b, num_inter, t, t, h_d, w_d)
        W = W.permute(0, 4, 5, 1, 2, 3).contiguous().view(b * hw_d * num_inter, t, t)

        # positive definite
        # W = torch.bmm(W, W.permute(0, 2, 1))
        W = torch.tanh(torch.bmm(W, W.permute(0, 2, 1)))
        W = W + (torch.eye(W.size(1)).cuda())

        "XTW & XTWX FOR NORMAL EQUATION"
        XTW = torch.bmm(X.permute(0, 2, 1), W)  # (b*hw_d)*num_inter, num_design, tile_size2
        XTWX = torch.bmm(XTW, X)  # (b*hw_d)*num_inter, num_design, num_design,

        "A = XTWX_sum FOR AX = B, LEAST SQUARE"
        XTWX_sum = torch.sum(XTWX.view(b * hw_d, num_inter, ch_de, ch_de), dim=1)
        # for stability, add the diagonal term
        XTWX_sum = XTWX_sum + (torch.eye(XTWX_sum.size(1)).cuda()) * self.epsilon  # A from Ax=B

        "REGRESSION FOR EACH CHANNEL"
        for ch in range(3):

            "Y FROM INPUT"
            Y = input[:, :, ch, :, :]  # b, t, h_d, w_d
            Y = self.unfold_and_padding(Y)  # b, t * num_inter, hw_d

            Y = Y.permute(0, 2, 1).contiguous().view(b * hw_d, t, num_inter)
            Y = Y.permute(0, 2, 1).contiguous().view(b * hw_d * num_inter, t).unsqueeze(2)

            XTWY = torch.bmm(XTW, Y)

            "B = XTWY_sum FOR AX = B, LEAST SQUARE"
            XTWY_sum = torch.sum(XTWY.view(b * hw_d, num_inter, ch_de, 1), dim=1)  # B from Ax=B

            "SOLVING LEAST SQUARE OF AX = B"
            para, _ = torch.solve(XTWY_sum, XTWX_sum)  # (b * hw_d), ch_de, 1

            "PREDICTION FOR DE NOISED COLOR BY PARA"
            out_1ch = torch.bmm(domain, para)  # (b*hw_d), t, 1
            out_1ch = out_1ch.view(b, h_d, w_d, t_de)
            out[:, :, ch, :, :] = out_1ch.permute(0, 3, 1, 2)

        return out


    def unfold_and_padding(self, x):
        """
        input : x (4D)
        output : Unfolded x
        feature #1 : unfolding을 하는 함수. padding mode를 조절할 수 있음.
        """
        kernel_length = self.length_inter
        if self.pad_mode > 0:
            pad = (kernel_length // 2, kernel_length // 2, kernel_length // 2, kernel_length // 2)
            if self.pad_mode == 1:
                x = nn.functional.pad(x, pad, mode='reflect')
            elif self.pad_mode == 2:
                x = nn.functional.pad(x, pad, mode='circular')
            else:
                x = nn.functional.pad(x, pad, mode='reflect')

            x_unfolded = F.unfold(x, kernel_length, padding=0)
        else:  # zero padding
            # automatically zero padding
            x_unfolded = F.unfold(x, kernel_length, padding=kernel_length//2)

        return x_unfolded

    def norm_in_prediction_window(self, design):
        """
                input : design (b*hw_d, inter_tile, t, ch_de)
                output : normalized design in terms of a prediction window
                feature #1 : 꼭 input 형태에 유의를 할 필요가 있음.
        """

        def min_max_norm(input):
            # input : b*hw_d, inter_tile, t, C
            a = input.dim()

            # min max
            if a == 4:
                min_input = torch.min(torch.min(torch.min(input, 1, True)[0], 2, True)[0], 3, True)[0]
                max_input = torch.max(torch.max(torch.max(input, 1, True)[0], 2, True)[0], 3, True)[0]
            else:
                min_input = torch.min(torch.min(input, 1, True)[0], 2, True)[0]
                max_input = torch.max(torch.max(input, 1, True)[0], 2, True)[0]

            return (input - min_input) / (max_input - min_input + 0.001)


        bhw_d, inter_tile, t, ch_de = design.size()

        # albedo
        design[:, :, :, 1:4] = min_max_norm(design[:, :, :, 1:4])

        # depth
        design[:, :, :, 4] = min_max_norm(design[:, :, :, 4])

        # normal
        design[:, :, :, 5:8] = min_max_norm(design[:, :, :, 5:8])

        # grid
        ch_grid = ch_de - 8
        for i in range(ch_grid):
            design[:, :, :, 8 + i] = min_max_norm(design[:, :, :, 8 + i])

        return design






class NPR_net_stack_v2(nn.Module):
    """
    네트워크 개요
    - NPR_net_stack_v1 구조를 활용함.
    - loss of unfolded data, norm in prediction window, choleski decomp
    - 새로운 데이터의 특성 상 loss함수를 따로 안에 정의를 함..

    새로 추가된 기능
    - loss of unfolded : unfolded 된 상태에서 loss를 구함.
    - overlapping : 각 prediction window의 크기 만큼 prediction and overlapping between them
    - norm in prediction window : WLR이나 다른 논문들 처럼 prediction window of regression에서 [0, 1]로 맞춤.
    - low weighted W : cholesky, LU 등 방법으로 W에 크기를 줄임.
    - refine the W generator : 먼저 R matrix -> unfolding을 위한 채널 늘리기.


    """

    def __init__(self, params, ch_in=10, kernel_size=3, tile_length=4, n_layers=12, length_inter_tile=5, epsilon=0.01,
                 pad_mode=0, unfolded_loss=True, norm_in_window=True, W_half=False):
        super(NPR_net_stack_v2, self).__init__()

        self.ch_in = ch_in
        self.k_size = kernel_size
        self.tile_length = tile_length

        self.tile_size = tile_length ** 2
        self.tile_size_stit = tile_length ** 2 + tile_length * 2

        self.epsilon = epsilon

        self.pad_mode = pad_mode  # 0: zero, 1: reflected, 2: circular

        self.length_inter = length_inter_tile
        self.inter_tile_num = int(length_inter_tile ** 2)

        self.no_stit_input = params["no_boundary_for_input"]
        self.no_stit_design = params["no_boundary_for_design"]

        "new features"
        # overlapping
        self.norm_in_window = norm_in_window
        self.unfolded_loss = unfolded_loss
        self.W_half = W_half

        # loss
        self.loss_fn = my_loss.loss_for_stit_v1(params['tile_length'], params["stitching_weights"], params['loss_type'])


        if self.no_stit_input:
            self.start_ch = ch_in * self.tile_size
            ft = self.tile_size
        else:
            self.start_ch = ch_in * self.tile_size_stit
            ft = self.tile_size_stit

        if self.W_half:
            self.numel_W_half = int((ft * (ft + 1)) / 2)
            self.final_ch = int(self.numel_W_half * self.inter_tile_num)
        else:
            self.final_ch = int((ft ** 2) * self.inter_tile_num)


        self.inter_ch = 1024

        self.layers = [nn.Conv2d(self.start_ch, self.inter_ch, kernel_size, padding=(kernel_size - 1) // 2),
                       nn.LeakyReLU()]

        for l in range(n_layers - 2):
            self.layers += [
                nn.Conv2d(self.inter_ch, self.inter_ch, kernel_size, padding=(kernel_size - 1) // 2),
                nn.LeakyReLU()
            ]

        self.feature_layers_feed = nn.Sequential(*self.layers)  # to get the feature

        self.layers_for_weights_feed = nn.Conv2d(self.inter_ch, self.final_ch, kernel_size,
                                            padding=(kernel_size - 1) // 2)

        # self.layers_for_weights_feed.weight.data.fill_(0.0)
        # self.layers_for_weights_feed.bias.data.fill_(0.0)


    def forward(self, input, design, gt, Flag_stack_result=False):
        """
        input : B T(tile_size) C_in H_d W_d
        design : B T(tile_size) C_de H_d W_d

        특이하게 gt를 넣어서 loss까지 한방에 계산을 하도록 함.
        output : resulting image and loss

        """
        "INITIAL SETTING"
        b = input.size(0)
        ch_in = input.size(2)  # 10 (color + g_buffer)
        ch_de = design.size(2)
        h_d, w_d = input.size(3), input.size(4)
        hw_d = h_d * w_d
        t_de = design.size(1)

        s = self.tile_length
        # t = self.tile_size_stit  # 24
        t = input.size(1)

        length_inter = self.length_inter  # length of inter tile
        num_inter = length_inter ** 2

        out = torch.zeros((b, t_de, 3, h_d, w_d), dtype=input.dtype, layout=input.layout,
                          device=input.device)

        if self.unfolded_loss:
            out_for_loss = torch.zeros((b * hw_d * num_inter, t_de, 3), dtype=input.dtype, layout=input.layout,
                          device=input.device)
        else:
            out_for_loss = torch.zeros((b * hw_d, t_de, 3), dtype=input.dtype, layout=input.layout,
                              device=input.device)

        "UNFOLD DESIGN MATRIX"
        design = design.view(b, t_de * ch_de, h_d, w_d)  # 5D -> 4D
        design = self.unfold_and_padding(design)  # b, t_de * ch_de * num_inter, hw_d


        "X FROM DESIGN MATRIX"
        design = design.permute(0, 2, 1).contiguous().view(b * hw_d, t_de * ch_de, num_inter)
        design = design.permute(0, 2, 1).contiguous().view(b * hw_d * num_inter, t_de * ch_de)

        design = design.view(b * hw_d, num_inter, t_de, ch_de)  # b * hw_d * num_inter, t_de, ch_de

        "v2 : norm in each prediction window according to channel"
        if self.norm_in_window:
            design = self.norm_in_prediction_window(design)
        # design = design.view(b * hw_d, num_inter, t_de, ch_de)

        # domain and X
        if self.unfolded_loss:
            domain = design.view(b * hw_d * num_inter, t_de, ch_de).contiguous()  # b * hw_d * num_inter, t_de, ch_de
        else:
            domain = design[:, num_inter//2, :, :].contiguous()  # b * hw_d, t_de, ch_de

        X = design[:, :, :t, :]
        X = X.view(b * hw_d * num_inter, t, ch_de)


        "GET THE FEATURE FROM FEATURE NETWORK"
        feature = self.feature_layers_feed(input.view(b, t * ch_in, h_d, w_d))

        "W FROM THE FEATURE"  # 여기서 메모리가 뻥튀기 됨.
        if self.W_half:
            W = self.layers_for_weights_feed(feature)
            W = W.permute(0, 2, 3, 1).contiguous().view((b * hw_d) * num_inter, self.numel_W_half)

            ### debug ###
            # W_half_debug = W.cpu().detach().numpy()
            ###       ###

            W = self.make_full_W_cholesky(W)

            ### debug ###
            # W_debug = W.cpu().detach().numpy()
            ###       ###

        else:
            W = torch.tanh(self.layers_for_weights_feed(feature))  # [-1, 1]
            W = W.view(b, num_inter, t, t, h_d, w_d)
            W = W.permute(0, 4, 5, 1, 2, 3).contiguous().view(b * hw_d * num_inter, t, t)

            # positive definite
            W = torch.bmm(W, W.permute(0, 2, 1))
            W = W + (torch.eye(W.size(1)).cuda())

            ### debug ###
            # W_debug = W.cpu().detach().numpy()
            ###       ###


        "XTW & XTWX FOR NORMAL EQUATION"
        XTW = torch.bmm(X.permute(0, 2, 1), W)  # (b*hw_d)*num_inter, num_design, tile_size2
        XTWX = torch.bmm(XTW, X)  # (b*hw_d)*num_inter, num_design, num_design,

        "A = XTWX_sum FOR AX = B, LEAST SQUARE"
        XTWX_sum = torch.sum(XTWX.view(b * hw_d, num_inter, ch_de, ch_de), dim=1)
        # for stability, add the diagonal term
        XTWX_sum = XTWX_sum + (torch.eye(XTWX_sum.size(1)).cuda()) * self.epsilon  # A from Ax=B


        "REGRESSION FOR EACH CHANNEL AND LOSS CALCULATION"

        "v2 : GT from gt for loss calculation"
        ref_for_loss = gt.view(b, t_de * 3, h_d, w_d)  # b, t, 3, h_d, w_d

        if self.unfolded_loss:
            ref_for_loss = self.unfold_and_padding(ref_for_loss)  # b, t * 3 * num_inter, hw_d
            ref_for_loss = ref_for_loss.permute(0, 2, 1).contiguous().view(b * hw_d, t_de * 3, num_inter)

            # (b * hw_d * num_inter), t, 3
            ref_for_loss = ref_for_loss.permute(0, 2, 1).contiguous().view(b * hw_d * num_inter, t_de, 3)
        else:
            # (b * hw_d), t, 3
            ref_for_loss = ref_for_loss.permute(0, 2, 3, 1).contiguous().view(b * hw_d, t_de, 3)

        for ch in range(3):
            "Y FROM INPUT"
            Y = input[:, :, ch, :, :]  # b, t, h_d, w_d
            Y = self.unfold_and_padding(Y)  # b, t * num_inter, hw_d

            Y = Y.permute(0, 2, 1).contiguous().view(b * hw_d, t, num_inter)
            Y = Y.permute(0, 2, 1).contiguous().view(b * hw_d * num_inter, t).unsqueeze(2)

            XTWY = torch.bmm(XTW, Y)

            "B = XTWY_sum FOR AX = B, LEAST SQUARE"
            XTWY_sum = torch.sum(XTWY.view(b * hw_d, num_inter, ch_de, 1), dim=1)  # B from Ax=B

            "SOLVING LEAST SQUARE OF AX = B"
            para, _ = torch.solve(XTWY_sum, XTWX_sum)  # (b * hw_d), ch_de, 1

            if self.unfolded_loss:
                # modifying para for applying to unfolded data  # (b * hw_d) * num_inter, ch_de, 1
                para = para.unsqueeze(1).repeat(1, num_inter, 1, 1).view((b * hw_d) * num_inter, ch_de, 1)

            "PREDICTION FOR DE NOISED COLOR BY PARA"
            out_1ch = torch.bmm(domain, para)  # (b*hw_d), t, 1 or (b*hw_d*num_inter), t, 1
            out_for_loss[:, :, ch] = out_1ch.squeeze(2)

            if Flag_stack_result:
                if self.unfolded_loss:
                    out_1ch = out_1ch.view(b, hw_d, num_inter, t_de)
                    out_1ch = out_1ch.permute(0, 3, 2, 1).contiguous()  # b, t, num_inter, hw_d
                    out_1ch = out_1ch.view(b, t_de * num_inter, hw_d)

                    ones = torch.ones_like(out_1ch)
                    ones_over = F.fold(ones, output_size=(h_d, w_d), kernel_size=length_inter,
                                       padding=length_inter // 2)

                    out_1ch_over = F.fold(out_1ch, output_size=(h_d, w_d), kernel_size=length_inter,
                                          padding=length_inter // 2) / ones_over

                    out[:, :, ch, :, :] = out_1ch_over

                else:
                    out_1ch = out_1ch.view(b, h_d, w_d, t_de)
                    out[:, :, ch, :, :] = out_1ch.permute(0, 3, 1, 2)

        "GET LOSS VALUE"
        Loss = self.loss_fn(out_for_loss, ref_for_loss)

        return out, Loss


    def unfold_and_padding(self, x):
        """
        input : x (4D)
        output : Unfolded x
        feature #1 : unfolding을 하는 함수. padding mode를 조절할 수 있음.
        """
        kernel_length = self.length_inter
        if self.pad_mode > 0:
            pad = (kernel_length // 2, kernel_length // 2, kernel_length // 2, kernel_length // 2)
            if self.pad_mode == 1:
                x = nn.functional.pad(x, pad, mode='reflect')
            elif self.pad_mode == 2:
                x = nn.functional.pad(x, pad, mode='circular')
            else:
                x = nn.functional.pad(x, pad, mode='reflect')

            x_unfolded = F.unfold(x, kernel_length, padding=0)
        elif self.pad_mode == 0:  # zero padding
            # automatically zero padding
            x_unfolded = F.unfold(x, kernel_length, padding=kernel_length//2)
        else:
            # image resolution gonna be reduced
            x_unfolded = F.unfold(x, kernel_length, padding=0)

        return x_unfolded

    def norm_in_prediction_window(self, design):
        """
                input : design (b*hw_d, inter_tile, t, ch_de)
                output : normalized design in terms of a prediction window
                feature #1 : 꼭 input 형태에 유의를 할 필요가 있음.
        """
        def min_max_norm(input):
            # input : b*hw_d, inter_tile, t, C
            a = input.dim()
            # min max
            if a == 4:
                min_input = torch.min(torch.min(torch.min(input, 1, True)[0], 2, True)[0], 3, True)[0]
                max_input = torch.max(torch.max(torch.max(input, 1, True)[0], 2, True)[0], 3, True)[0]
            else:
                min_input = torch.min(torch.min(input, 1, True)[0], 2, True)[0]
                max_input = torch.max(torch.max(input, 1, True)[0], 2, True)[0]

            return (input - min_input) / (max_input - min_input + 0.001)


        bhw_d, inter_tile, t, ch_de = design.size()

        # albedo
        design[:, :, :, 1:4] = min_max_norm(design[:, :, :, 1:4])

        # depth
        design[:, :, :, 4] = min_max_norm(design[:, :, :, 4])

        # normal
        design[:, :, :, 5:8] = min_max_norm(design[:, :, :, 5:8])

        # grid
        ch_grid = ch_de - 8
        for i in range(ch_grid):
            design[:, :, :, 8 + i] = min_max_norm(design[:, :, :, 8 + i])

        return design

    def make_full_W(self, W_half, epsilon=0.01):
        "기존 것과는 다르게 [-1, 1] 사이로 하지 않음. covariance 형태로 갈 예정"
        "또한 W_half 형태도 바꿀 예정 : (b*hw//tile_size)*num_inter, num_W_half"

        tile_size = self.tile_size

        W_full = torch.zeros((W_half.size(0), tile_size, tile_size), dtype=W_half.dtype,
                             layout=W_half.layout, device=W_half.device)

        index = 0
        for i in range(tile_size):
            diag = torch.sigmoid(W_half[:, index])
            W_full[:, i, i] = diag + epsilon

            # no_diag = torch.tanh(W_half[:, (index + 1):(index + 1) + (tile_size - 1 - i)]) * diag.unsqueeze(1)
            if not i==tile_size-1:
                no_diag = torch.tanh(W_half[:, (index + 1):(index + 1) + (tile_size - 1 - i)])
                # no_diag_absmax = torch.max(torch.abs(no_diag), 1, True)[0]
                # W_full[:, i, i+1:] = (no_diag / (no_diag_absmax + 0.0001)) * diag.unsqueeze(1)
                # W_full[:, i+1:, i] = (no_diag / (no_diag_absmax + 0.0001)) * diag.unsqueeze(1)
                W_full[:, i, i + 1:] = no_diag
                W_full[:, i + 1:, i] = no_diag

            index += 1 + (tile_size - 1 - i)

        ### debug ###
        # W_full_debug = W_full.cpu().detach().numpy()
        ###       ###

        return W_full

    def make_full_W_cholesky(self, W_half, epsilon=0.01):
        "기존 것과는 다르게 [-1, 1] 사이로 하지 않음. covariance 형태로 갈 예정"
        "또한 W_half 형태도 바꿀 예정 : (b*hw//tile_size)*num_inter, num_W_half"
        tile_size = self.tile_size

        U = torch.zeros((W_half.size(0), tile_size, tile_size), dtype=W_half.dtype,
                             layout=W_half.layout, device=W_half.device)
        index = 0
        for i in range(tile_size):

            diag = torch.sigmoid(W_half[:, index])
            U[:, i, i] = diag + epsilon

            no_diag = torch.tanh(W_half[:, (index + 1):(index + 1) + (tile_size - 1 - i)])
            U[:, i, i+1:] = no_diag
            U[:, i + 1:, i] = no_diag
            index += 1 + (tile_size - 1 - i)

        ### debug ###
        # U_debug = U.cpu().detach().numpy()
        ###       ###

        W_full = torch.bmm(U.permute(0, 2, 1), U)

        ### debug ###
        # W_full_debug = W_full.cpu().detach().numpy()
        ###       ###

        return W_full




class NPR_net_img_v1(nn.Module):
    """
    네트워크 개요
    - input은 full res이고 design은 stack으로 들어오게 됨.
    - "NPR_net_stack_v1"의 구조를 활용함.
    - stitching은 domain의 크기를 조절하는 것으로 해서 구현할 예정.
    - 네트워크도 2가지로 만들 예정이다.
        - 1. feature network = full_res 이미지를 두고 feature를 뽑는 네트워크
        - 2. weight network = 타일의 크기에 맞게 downscale된 feature를 대상으로 weight을 뽑는 네트워크. 

    """

    def __init__(self, ch_in=10, kernel_size=3, tile_length=4, n_layers=20, length_inter_tile=7, epsilon=0.01,
                 pad_mode=0, no_stit_design=True):
        super(NPR_net_img_v1, self).__init__()

        self.channels_in = ch_in
        self.k_size = kernel_size
        self.tile_length = tile_length

        self.tile_size = tile_length ** 2
        self.tile_size_stit = tile_length ** 2 + tile_length * 2

        self.epsilon = epsilon

        self.pad_mode = pad_mode  # 0: zero, 1: reflected, 2: circular

        self.length_inter = length_inter_tile
        self.inter_tile_num = int(length_inter_tile ** 2)

        self.no_stit_design = no_stit_design
        
        # full res network
        self.inter_ch_for_full_res = 100
        self.layers_for_feature = [nn.Conv2d(ch_in, self.inter_ch_for_full_res, kernel_size, padding=(kernel_size - 1) // 2),
                                        nn.LeakyReLU()]

        for l in range(n_layers - 4 - length_inter_tile//2):
            self.layers_for_feature += [
                nn.Conv2d(self.inter_ch_for_full_res, self.inter_ch_for_full_res, kernel_size, padding=(kernel_size - 1) // 2),
                nn.LeakyReLU()
            ]
        self.feature_layers_feed = nn.Sequential(*self.layers_for_feature)  # to get the feature
        
        # down scaled network : 지정된 tile length에 따라서 down scale을 하게 된다.
        self.final_ch = int((self.tile_size ** 2) * self.inter_tile_num)

        self.layers_for_weight = [
            # down conv according to the tile_length
            nn.Conv2d(self.inter_ch_for_full_res, 1024, tile_length, stride=tile_length, padding=0),
            nn.LeakyReLU(),
        ]

        for l in range(length_inter_tile//2):
            self.layers_for_weight += [
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(),
            ]

        self.layers_for_weight += [nn.Conv2d(1024, self.final_ch//2, 1, padding=0),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(self.final_ch//2, self.final_ch, 1, padding=0)]

        self.layers_for_weights_feed = nn.Sequential(*self.layers_for_weight)

        eee = self.layers_for_weight[-1]
        # eee.weight.data.fill_(0.0)
        # eee.bias.data.fill_(0.0)



    def forward(self, input, design):
        """
        input : B T(tile_size) C_in H_d W_d
        design : B T(tile_size) C_de H_d W_d
        """

        "INITIAL SETTING"
        b = input.size(0)
        ch_in = input.size(2)  # 10 (color + g_buffer)
        ch_de = design.size(2)
        t_de = design.size(1)
        h_d, w_d = input.size(3), input.size(4)
        hw_d = h_d * w_d

        s = self.tile_length
        t = self.tile_size  # 16
        length_inter = self.length_inter  # length of inter tile
        num_inter = length_inter ** 2

        # domain for prediction using parameters which is the result of least square
        domain = design.permute(0, 3, 4, 1, 2).contiguous().view(b * hw_d, t_de, ch_de)

        if not self.no_stit_design:
            design = design[:, :s**2, :, :, :].contiguous()

        out = torch.zeros((b, t_de, 3, h_d, w_d), dtype=input.dtype, layout=input.layout,
                          device=input.device)

        "UNFOLD DESIGN MATRIX"
        design = design.view(b, t * ch_de, h_d, w_d)  # 5D -> 4D
        design = self.unfold_and_padding(design)

        "X FROM DESIGN MATRIX"
        design = design.permute(0, 2, 1).contiguous().view(b * hw_d, t * ch_de, num_inter)
        design = design.permute(0, 2, 1).contiguous().view(b * hw_d * num_inter, t * ch_de)
        X = design.view(-1, t, ch_de)

        "GET THE FEATURE FROM FEATURE NETWORK"
        input_full = self.make_full_res_img(input)
        feature = self.feature_layers_feed(input_full)

        "W FROM THE FEATURE"  # 여기서 메모리가 뻥튀기 됨.
        W = torch.tanh(self.layers_for_weights_feed(feature))
        # W = self.layers_for_weights_feed(feature)
        W = W.view(b, num_inter, t, t, h_d, w_d)
        W = W.permute(0, 4, 5, 1, 2, 3).contiguous().view(b * hw_d * num_inter, t, t)

        # positive definite
        W = torch.bmm(W, W.permute(0, 2, 1))
        W = W + (torch.eye(W.size(1)).cuda())

        "XTW & XTWX FOR NORMAL EQUATION"
        XTW = torch.bmm(X.permute(0, 2, 1), W)  # (b*hw_d)*num_inter, num_design, tile_size2
        XTWX = torch.bmm(XTW, X)  # (b*hw_d)*num_inter, num_design, num_design,

        "A = XTWX_sum FOR AX = B, LEAST SQUARE"
        XTWX_sum = torch.sum(XTWX.view(b * hw_d, num_inter, ch_de, ch_de), dim=1)
        # for stability, add the diagonal term
        XTWX_sum = XTWX_sum + (torch.eye(XTWX_sum.size(1)).cuda()) * self.epsilon  # A from Ax=B

        "REGRESSION FOR EACH CHANNEL"
        for ch in range(3):
            "Y FROM INPUT"
            Y = input[:, :, ch, :, :]  # b, t, h_d, w_d
            Y = self.unfold_and_padding(Y)  # b, t * num_inter, hw_d

            Y = Y.permute(0, 2, 1).contiguous().view(b * hw_d, t, num_inter)
            Y = Y.permute(0, 2, 1).contiguous().view(b * hw_d * num_inter, t).unsqueeze(2)

            XTWY = torch.bmm(XTW, Y)

            "B = XTWY_sum FOR AX = B, LEAST SQUARE"
            XTWY_sum = torch.sum(XTWY.view(b * hw_d, num_inter, ch_de, 1), dim=1)  # B from Ax=B

            "SOLVING LEAST SQUARE OF AX = B"
            para, _ = torch.solve(XTWY_sum, XTWX_sum)  # (b * hw_d), ch_de, 1

            "PREDICTION FOR DE NOISED COLOR BY PARA"
            out_1ch = torch.bmm(domain, para)  # (b*hw_d), t, 1
            out_1ch = out_1ch.view(b, h_d, w_d, t_de)
            out[:, :, ch, :, :] = out_1ch.permute(0, 3, 1, 2)

        return out


    def unfold_and_padding(self, x):
        """
        input : x (4D)
        output : Unfolded x
        feature #1 : unfolding을 하는 함수. padding mode를 조절할 수 있음.
        """
        kernel_length = self.length_inter
        if self.pad_mode > 0:
            pad = (kernel_length // 2, kernel_length // 2, kernel_length // 2, kernel_length // 2)
            if self.pad_mode == 1:
                x = nn.functional.pad(x, pad, mode='reflect')
            elif self.pad_mode == 2:
                x = nn.functional.pad(x, pad, mode='circular')
            else:
                x = nn.functional.pad(x, pad, mode='reflect')

            x_unfolded = F.unfold(x, kernel_length, padding=0)
        else:  # zero padding
            # automatically zero padding
            x_unfolded = F.unfold(x, kernel_length, padding=kernel_length//2)

        return x_unfolded


    def stack_elements(self, input):
        "input: 3ch tensor"
        b, ch, h, w = input.size()
        s = int(self.tile_length)
        tile_size = s ** 2

        out = torch.zeros((b, tile_size, ch, h // s, w // s), dtype=input.dtype, layout=input.layout,
                          device=input.device)

        for index in range(tile_size):
            i = index // s
            j = index % s
            out[:, j + s * i, :, :, :] = input[:, :, i::s, j::s]

        return out

    def make_full_res_img(self, out_stack):
        h, w = out_stack.size(3), out_stack.size(4)
        ch = out_stack.size(2)
        b = out_stack.size(0)

        # out_stack : (b, 3, h // s, w // s, tile_size)
        s = int(self.tile_length)

        full_res_img = torch.zeros((b, ch, h * s, w * s), dtype=out_stack.dtype, layout=out_stack.layout,
                                   device=out_stack.device)

        for index in range(s ** 2):
            i = index // s
            j = index % s
            full_res_img[:, :, i::s, j::s] = out_stack[:, j + s * i, :, :, :]

        return full_res_img