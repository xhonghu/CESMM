import torch
import torch.nn as nn
import math
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, channel, joint_num, time_len):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len
        pos_list = []
        for t in range(self.time_len):
            for j_id in range(self.joint_num):
                pos_list.append(j_id)
        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :, :x.size(2)]
        return x


class TemporalPooling4D(nn.Module):
    def __init__(self, input_size, kernel_size=2):
        super(TemporalPooling4D, self).__init__()
        self.kernel_size = kernel_size

        self.ConvE = self._build_conv_block(input_size)
        self.ConvO = self._build_conv_block(input_size)

        self.ConvE[3].weight.data.fill_(0.0)
        self.ConvO[3].weight.data.fill_(0.0)

    def _build_conv_block(self, input_size):
        return nn.Sequential(
            nn.Conv2d(input_size, input_size, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), groups=input_size),
            nn.BatchNorm2d(input_size),
            nn.LeakyReLU(0.1),
            nn.Conv2d(input_size, input_size, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(input_size)
        )

    def forward(self, x):
        Xe = x[:, :, ::self.kernel_size, :]
        Xo = x[:, :, 1::self.kernel_size, :]
        Ee = Xe + self.ConvO(Xo)
        Eo = Xo + self.ConvE(Xe)
        merged_output = Ee + Eo
        
        return merged_output


class MultiGranularityTemporalConv(nn.Module):
    def __init__(self, channels,  t_kernel=3, stride=1):
        super(MultiGranularityTemporalConv, self).__init__()
        reduction_channel = channels // 4

        self.down_conv =  nn.Sequential(
            nn.Conv2d(channels, reduction_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_channel),
            nn.LeakyReLU(0.1)
            )
        
        padd1 = (t_kernel - 1) // 2
        # 分支1: 捕捉短期依赖
        self.branch1 = nn.Sequential(
            nn.Conv2d(reduction_channel, reduction_channel, kernel_size=(t_kernel, 1), stride=(stride, 1), padding=(padd1, 0), groups=reduction_channel),
            nn.BatchNorm2d(reduction_channel),
            nn.LeakyReLU(0.1),
            nn.Conv2d(reduction_channel, reduction_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_channel)
        )
        padd2 = (2 * (t_kernel - 1)) // 2
        # 分支2: 使用空洞卷积捕捉中期依赖
        self.branch2 = nn.Sequential(
            nn.Conv2d(reduction_channel, reduction_channel, kernel_size=(t_kernel, 1), stride=(stride, 1), padding=(padd2, 0), dilation=(2, 1), groups=reduction_channel),
            nn.BatchNorm2d(reduction_channel),
            nn.LeakyReLU(0.1),
            nn.Conv2d(reduction_channel, reduction_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_channel)
        )
        padd3 = (3 * (t_kernel - 1)) // 2
        # 分支3: 使用更大的空洞卷积捕捉长期依赖
        self.branch3 = nn.Sequential(
            nn.Conv2d(reduction_channel, reduction_channel, kernel_size=(t_kernel, 1), stride=(stride, 1), padding=(padd3, 0), dilation=(3, 1), groups=reduction_channel),
            nn.BatchNorm2d(reduction_channel),
            nn.LeakyReLU(0.1),
            nn.Conv2d(reduction_channel, reduction_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_channel)
        )

        self.conv_back =  nn.Sequential(
            nn.Conv2d(reduction_channel, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
            )

        if stride != 1:
            self.downt = TemporalPooling4D(input_size=channels, kernel_size=2)
        else:
            self.downt = lambda x: x
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        res = self.downt(x)
        y = self.down_conv(x)
        out1 = self.branch1(y)
        out2 = self.branch2(y)
        out3 = self.branch3(y)
        out = self.conv_back(out1 + out2 + out3)
        return self.relu(out + res)


class PartBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, num_subset=2, num_node=27, num_frame=400,
                 stride=1, attentiondrop=0.1):
        super(PartBlock, self).__init__()
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset

        atts = torch.zeros((1, num_subset, num_node, num_node))
        self.register_buffer('atts', atts)
        self.pes = PositionalEncoding(in_channels, num_node, num_frame)
        self.ff_nets = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
        )

        self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
        self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
        self.attention0s = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node,
                                        requires_grad=True)

        self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels),
        )
        
        self.MTP = MultiGranularityTemporalConv(channels=out_channels, stride=stride)

        if in_channels != out_channels:
            self.downs1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downs1 = lambda x: x
            self.downs2 = lambda x: x

        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(attentiondrop)

    def forward(self, x):
        N, C, T, V = x.size()
        attention = self.atts
        y = self.pes(x)
        q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2,
                            dim=1)
        attention = attention + self.tan(torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)) * self.alphas
        attention = attention + self.attention0s.repeat(N, 1, 1, 1)
        attention = self.drop(attention)
        y = torch.einsum('nctu,nsuv->nsctv', [x, attention]).contiguous() \
            .view(N, self.num_subset * self.in_channels, T, V)
        y = self.out_nets(y)
        y = self.relu(self.downs1(x) + y)
        y = self.ff_nets(y)
        y = self.relu(self.downs2(x) + y)
        z = self.MTP(y)
        return z


class CrossPartFusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.global_context_mlp = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels)
        )

    def forward(self, contour, eyebrow, nose, eye, mouth, left, right):
        contour_pool = contour.mean(dim=-1) # (N, C, T)
        eyebrow_pool = eyebrow.mean(dim=-1)
        nose_pool = nose.mean(dim=-1)
        eye_pool = eye.mean(dim=-1)
        mouth_pool = mouth.mean(dim=-1)
        left_pool = left.mean(dim=-1)
        right_pool = right.mean(dim=-1)

        global_context = (contour_pool + eyebrow_pool + nose_pool + eye_pool + mouth_pool + left_pool + right_pool) / 7.0 # (N, C, T)

        refined_context = self.global_context_mlp(global_context.permute(0, 2, 1)).permute(0, 2, 1)

        contour = contour +  refined_context.unsqueeze(-1)
        eyebrow = eyebrow +  refined_context.unsqueeze(-1)
        nose = nose +  refined_context.unsqueeze(-1)
        eye = eye +  refined_context.unsqueeze(-1)
        mouth = mouth +  refined_context.unsqueeze(-1)
        left = left +  refined_context.unsqueeze(-1)
        right = right +  refined_context.unsqueeze(-1)

        return contour, eyebrow, nose, eye, mouth, left, right


class CESMM(nn.Module):
    def __init__(self, num_frame=400,
                 num_subset=6, input_type='keypoint',
                 num_channel=2):
        super(CESMM, self).__init__()
        self.input_type = input_type
        in_channels = 64
        Network = [[64, 64, 16, 1], [64, 64, 16, 1],
                  [64, 128, 32, 2], [128, 128, 32, 1],
                  [128, 256, 64, 1], [256, 256, 64, 1],
                  [256, 256, 64, 2], [256, 256, 64, 1]]
        self.num_frame = num_frame
        self.face_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.left_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.right_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.contour_graph_layers = nn.ModuleList()
        self.eyebrow_graph_layers = nn.ModuleList()
        self.nose_graph_layers = nn.ModuleList()
        self.eye_graph_layers = nn.ModuleList()
        self.mouth_graph_layers = nn.ModuleList()
        self.left_graph_layers = nn.ModuleList()
        self.right_graph_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()

        if  "keypoint" in self.input_type:
            num_bones = {
                'contour': 17,
                'eyebrow': 10,
                'nose': 9,
                'eye': 12,
                'mouth': 20,
                'left': 24,
                'right': 24,
            }
        else:
            num_bones = {
                'contour': 17 - 1,
                'eyebrow': 10 - 1,
                'nose': 9 - 1,
                'eye': 12 - 1,
                'mouth': 20 - 1,
                'left': 24 - 1,
                'right': 24 - 1,
            }
        
        for index, (in_channels, out_channels, inter_channels, stride) in enumerate(Network):
            self.contour_graph_layers.append(
                PartBlock(in_channels, out_channels, inter_channels, stride=stride,  num_node=num_bones['contour'],
                                 num_frame=num_frame, num_subset=num_subset))
            self.eyebrow_graph_layers.append(
                PartBlock(in_channels, out_channels, inter_channels, stride=stride,  num_node=num_bones['eyebrow'],
                                 num_frame=num_frame, num_subset=num_subset))
            self.nose_graph_layers.append(
                PartBlock(in_channels, out_channels, inter_channels, stride=stride,  num_node=num_bones['nose'],
                                 num_frame=num_frame, num_subset=num_subset))
            self.eye_graph_layers.append(
                PartBlock(in_channels, out_channels, inter_channels, stride=stride,  num_node=num_bones['eye'],
                                 num_frame=num_frame, num_subset=num_subset))
            self.mouth_graph_layers.append(
                PartBlock(in_channels, out_channels, inter_channels, stride=stride,  num_node=num_bones['mouth'],
                                 num_frame=num_frame, num_subset=num_subset))
            self.left_graph_layers.append(
                PartBlock(in_channels, out_channels, inter_channels, stride=stride,  num_node=num_bones['left'],
                                 num_frame=num_frame, num_subset=num_subset))
            self.right_graph_layers.append(
                PartBlock(in_channels, out_channels, inter_channels, stride=stride,  num_node=num_bones['right'],
                                 num_frame=num_frame, num_subset=num_subset))
            self.fusion_layers.append(
                CrossPartFusionBlock(out_channels)
            )
            num_frame = int(num_frame / stride + 0.5)

    def forward(self,x):
        N, C, T, V = x.shape
        x = x.permute(0, 1, 2, 3).contiguous().view(N, C, T, V)

        face_joints = self.face_input_map(x[:, :, :, list(range(23, 91))])   #68
        contour_joints = face_joints[:, :, :, list(range(0, 17))]
        eyebrow_joints = face_joints[:, :, :, list(range(17, 27))]
        nose_joints = face_joints[:, :, :, list(range(27, 36))]
        eye_joints = face_joints[:, :, :, list(range(36, 48))]
        mouth_joints = face_joints[:, :, :, list(range(48, 68))]
        left_joints = self.left_input_map(x[:, :, :, [6, 8, 10]+list(range(112, 133))])  #24
        right_joints = self.right_input_map(x[:, :, :, [5, 7, 9]+list(range(91, 112))])  #24

        if  "keypoint" in self.input_type:
            contour = contour_joints
            eyebrow = eyebrow_joints
            nose = nose_joints
            eye = eye_joints
            mouth = mouth_joints
            left = left_joints
            right = right_joints
        else:
            contour = contour_joints[:, :, :, 1:] - contour_joints[:, :, :, :-1]
            eyebrow = eyebrow_joints[:, :, :, 1:] - eyebrow_joints[:, :, :, :-1]
            nose = nose_joints[:, :, :, 1:] - nose_joints[:, :, :, :-1]
            eye = eye_joints[:, :, :, 1:] - eye_joints[:, :, :, :-1]
            mouth = mouth_joints[:, :, :, 1:] - mouth_joints[:, :, :, :-1]
            left = left_joints[:, :, :, 1:] - left_joints[:, :, :, :-1]
            right = right_joints[:, :, :, 1:] - right_joints[:, :, :, :-1]
        
        if "motion" in self.input_type:
            contour = torch.cat((torch.zeros_like(contour[:, :, :1, :]), contour[:, :, 1:, :] - contour[:, :, :-1, :]), dim=2)
            eyebrow = torch.cat((torch.zeros_like(eyebrow[:, :, :1, :]), eyebrow[:, :, 1:, :] - eyebrow[:, :, :-1, :]), dim=2)
            nose = torch.cat((torch.zeros_like(nose[:, :, :1, :]), nose[:, :, 1:, :] - nose[:, :, :-1, :]), dim=2)
            eye = torch.cat((torch.zeros_like(eye[:, :, :1, :]), eye[:, :, 1:, :] - eye[:, :, :-1, :]), dim=2)
            mouth = torch.cat((torch.zeros_like(mouth[:, :, :1, :]), mouth[:, :, 1:, :] - mouth[:, :, :-1, :]), dim=2)
            left = torch.cat((torch.zeros_like(left[:, :, :1, :]), left[:, :, 1:, :] - left[:, :, :-1, :]), dim=2)
            right = torch.cat((torch.zeros_like(right[:, :, :1, :]), right[:, :, 1:, :] - right[:, :, :-1, :]), dim=2)

        for i in range(len(self.contour_graph_layers)):
            contour = self.contour_graph_layers[i](contour)
            eyebrow = self.eyebrow_graph_layers[i](eyebrow)
            mouth = self.mouth_graph_layers[i](mouth)
            nose = self.nose_graph_layers[i](nose)
            eye = self.eye_graph_layers[i](eye)
            left = self.left_graph_layers[i](left)
            right = self.right_graph_layers[i](right)
            contour, eyebrow, nose, eye, mouth, left, right = self.fusion_layers[i](contour, eyebrow, nose, eye, mouth, left, right)

        contour = contour.permute(0, 2, 1, 3).contiguous()
        eyebrow = eyebrow.permute(0, 2, 1, 3).contiguous()
        mouth = mouth.permute(0, 2, 1, 3).contiguous()    
        nose = nose.permute(0, 2, 1, 3).contiguous()
        eye = eye.permute(0, 2, 1, 3).contiguous()
        left = left.permute(0, 2, 1, 3).contiguous()
        right = right.permute(0, 2, 1, 3).contiguous()

        face = torch.cat([contour, eyebrow, mouth, nose, eye], dim=-1)
        hand = torch.cat([left, right], dim=-1)

        face = face.mean(3)
        hand = hand.mean(3)
        mouth = mouth.mean(3)
        left = left.mean(3)
        right = right.mean(3)

        output = torch.cat([face, hand, mouth, left, right], dim=-1).permute(0, 2, 1)
        return output