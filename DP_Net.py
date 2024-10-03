""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def calculate_distance_tensor(points):
    # 使用广播机制计算所有点对之间的欧氏距离
    points = points.float()
    """
    具体来说，假设 points 是一个大小为 (N, D) 的张量，其中 N 是点的数量，D 是每个点的维度，
    那么 torch.cdist(points, points) 会返回一个大小为 (N, N) 的距离矩阵 dist_matrix，其中 dist_matrix[i, j] 表示第 i 个点和第 j 个点之间的欧氏距离。
    """
    dist_matrix = torch.cdist(points, points)
    return dist_matrix

def Attention_Radius(H, W):
    # 使用PyTorch张量生成坐标列表
    """
    #torch.arange(H, 0, -1, device=device) 生成 y 坐标的序列 torch.arange(1, W + 1, device=device) 生成 x 坐标的序列。
    torch.meshgrid 生成网格坐标矩阵 y_coords 和 x_coords

    """
    y_coords, x_coords = torch.meshgrid(torch.arange(H, 0, -1), torch.arange(1, W + 1))
    #torch.stack 将 x_coords 和 y_coords 展开并组合成每个点的 (x, y) 坐标对。
    coordinates = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1)
    # 计算距离矩阵
    distance_matrix = calculate_distance_tensor(coordinates)
    return distance_matrix

def Weight_Mapping(data, new_min=0.5, new_max=2):
    data = data.to('cuda')
    new_min = torch.tensor(new_min, device='cuda')
    new_max = torch.tensor(new_max, device='cuda')

    original_min = data.min()
    original_max = data.max()

    scaled_data = new_max - (data - original_min) * (new_max - new_min) / (original_max - original_min)
    return scaled_data

class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.conv_phi = nn.Conv2d(channel, channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(channel, channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_g = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmaxrw = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        # [N, C, H , W]
        #[1, 1024, 16, 16]
        b, c, h, w = x.size()
        #[1, 1024, 16, 16]->[1, 1024, 256]
        #[N, C, H , W]->[N, C, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        #[1, 1024, 16, 16]->[1, 256, 1024]
        #[N, C, H , W]->[N, H * W, C]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        #[1, 1024, 16, 16]->[1, 256, 1024]
        #[N, C, H , W]->[N, H * W, C]
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        #[1, 256, 1024] [1, 1024, 256] -> [1, 256, 256]
        #[N, H * W, C] [N, C, H * W]->[N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        #到了这一步的mul_theta_phi相当于是本来生成好的注意力权重map
        """
        开始生成半径权重图
        """
        radius = Attention_Radius(h, w)
        radius = Weight_Mapping(radius,new_min=0.5,new_max=2)
        mul_theta_phi = radius * mul_theta_phi
        mul_theta_phi = self.softmaxrw(mul_theta_phi)
        #[1, 256, 256] [1, 256, 1024] -> [1, 256, 1024]
        #[N, H * W, H * W] [N, H * W, C] -> [N, H * W, C]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        #[1, 256, 1024]->[1, 1024, 16, 16]
        #[N, H * W, C]->[N, C, H , W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, c, h, w)
        mask = mul_theta_phi_g
        out = mask + x # 残差连接
        return out


class FeatureFsuion(nn.Module):
    def __init__(self, channel):
        super(FeatureFsuion, self).__init__()
        self.channel_mlp1 = nn.Sequential(
            nn.Linear(channel, channel // 2),
            nn.ReLU(),
            nn.Linear(channel // 2, channel),
        )

        self.sigmoid = nn.Sigmoid()
        self.spatial_conv1_2 = nn.Conv2d(in_channels=channel,
                                         out_channels=1,
                                         kernel_size=(1, 1),
                                         stride=(1, 1))

    def forward(self, L,G):
        # [N, C, H , W]
        """
        全局特征指导局部特征
        全局特征生成通道权重
        """
        residual_L = L
        # 全局平均池化
        # [8, 512, 32, 32]->[8, 512, 1, 1]
        G_avg_pool = F.adaptive_avg_pool2d(G, (1, 1))

        # [8, 512, 1, 1]->[8, 1, 1, 512]
        G_avg_pool = G_avg_pool.permute(0, 2, 3, 1)
        # MLP
        # [8, 1, 1, 512] -> [8, 1, 1, 512]
        G_avg_pool = self.channel_mlp1(G_avg_pool)
        # 全局最大池化
        # [8, 512, 32, 32]->[8, 512, 1, 1]
        G_max_pool = F.adaptive_max_pool2d(G, (1, 1))
        # [8, 512, 1, 1]->[8, 1, 1,512]
        G_max_pool = G_max_pool.permute(0, 2, 3, 1)
        # MLP
        # [8, 1, 1,512]->[8, 512,1,1]
        G_max_pool = self.channel_mlp1(G_max_pool)
        # [8, 512,1,1]
        G_weight = G_max_pool + G_avg_pool
        """全局特征指导局部特征"""
        G_weight = G_weight.permute(0, 3, 1, 2)
        G_weight = self.sigmoid(G_weight)
        #局部特征与全局权重相乘
        # [1, 64, 16, 16, 16] * [1,64,1,1,1]->[1, 64, 16, 16, 16]
        L = L * G_weight

        """全局特征生成空间注意力图"""
        G_s_weight = self.spatial_conv1_2(G)
        G_s_weight = self.sigmoid(G_s_weight)

        """全局特征指导空间特征"""
        L = L * G_s_weight

        L = L + residual_L



        return L


class DPNet(nn.Module):
    def __init__(self, in_channels=3,channels=64,out_channels=1,  bilinear=False):
        super(DPNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        #上方路径实例化
        self.inc_1 = (DoubleConv(in_channels, channels))

        self.down1_2 = (Down(channels, 2*channels))

        self.down1_3 = (Down(2*channels, 4*channels))

        self.down1_4 = (Down(4*channels, 8*channels))
        factor = 2 if bilinear else 1
        self.down1_5 = (Down(8*channels, 16*channels // factor))



        #下方路径实例化

        self.down2_4 = (Down(4 * channels, 8 * channels))
        self.non_local_attention4 = NonLocalBlock(8 * channels)
        #融合两层编码器的特征
        self.fusion4 = FeatureFsuion(8 * channels)

        factor = 2 if bilinear else 1
        self.down2_5 = (Down(8 * channels, 16 * channels // factor))
        self.non_local_attention5 = NonLocalBlock(16 * channels // factor)
        self.fusion5 = FeatureFsuion(16 * channels // factor)

        self.up2_1 = (Up(16 * channels, 8 * channels // factor, bilinear))

        self.up2_2 = (Up(8 * channels, 4 * channels // factor, bilinear))

        self.up2_3 = (Up(4 * channels, 2 * channels // factor, bilinear))

        self.up2_4 = (Up(2 * channels, channels, bilinear))



        self.outc = (OutConv(channels, out_channels))

    def forward(self, x):
        """
        第一条编码器
        """
        #[1,3,256,256]->[8, 64, 256, 256]
        x1 = self.inc_1(x)

        #[8, 64, 256, 256] -> [8, 128, 128, 128]
        x2 = self.down1_2(x1)

        #[8, 128, 128, 128] -> [8, 256, 64, 64]
        x3 = self.down1_3(x2)


        #[8, 256, 64, 64]->[8, 512, 32, 32]
        x4 = self.down1_4(x3)

        #[8, 512, 32, 32] -> [8, 1024, 16, 16]
        x5 = self.down1_5(x4)


        """
        第二条编码器
        """

        # [8, 256, 64, 64]->[8, 512, 32, 32]
        y4 = self.down2_4(x3)
        y4 = self.non_local_attention4(y4)

        # [8, 512, 32, 32] -> [8, 1024, 16, 16]
        y5 = self.down2_5(y4)
        y5 = self.non_local_attention5(y5)

        """
        两条编码器的各级特征融合
        """
        x4 = self.fusion4(x4, y4)
        x5 = self.fusion5(x5, y5)
        """
        解码器
        """
        # x5 [8, 1024, 16, 16] x4 [8, 512, 32, 32]  x [8, 512, 32, 32]
        xu = self.up2_1(x5, x4)

        # [8, 512, 32, 32] [8, 256, 64, 64]->[8, 256, 64, 64]
        xu = self.up2_2(xu, x3)

        # [8, 256, 64, 64] [8, 128, 128, 128]->[8, 128, 128, 128]
        xu = self.up2_3(xu, x2)

        # [8, 128, 128, 128] [8, 64, 256, 256]->[8, 64, 256, 256]
        xu = self.up2_4(xu, x1)


        logits = self.outc(xu)
        return logits

if __name__ == '__main__':
    image = torch.randn(1, 3, 256, 256)
    model = DPNet(in_channels=3,channels=64,out_channels=1,  bilinear=False)
    print(model(image).shape)
