import torch
import torch.nn as nn


# ========= UNet模型定义 =========
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ImprovedDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0, max_pooling=True):
        super(ImprovedDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2) if max_pooling else None
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None

        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.residual_conv:
            identity = self.residual_conv(identity)
        x = self.relu(x + identity)

        if self.dropout:
            x = self.dropout(x)
        skip = x
        if self.maxpool:
            x = self.maxpool(x)
        return x, skip


class ImprovedUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImprovedUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = AttentionGate(out_channels, out_channels, out_channels // 2)

        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.attention(x, skip)
        x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.relu(x)

        return x


class ImprovedUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, n_filters=64):
        super(ImprovedUNet, self).__init__()

        self.down1 = ImprovedDownBlock(n_channels, n_filters)
        self.down2 = ImprovedDownBlock(n_filters, n_filters * 2)
        self.down3 = ImprovedDownBlock(n_filters * 2, n_filters * 4)
        self.down4 = ImprovedDownBlock(n_filters * 4, n_filters * 8)
        self.down5 = ImprovedDownBlock(n_filters * 8, n_filters * 16)

        self.bottleneck = ImprovedDownBlock(n_filters * 16, n_filters * 32, dropout_prob=0.5, max_pooling=False)

        self.up1 = ImprovedUpBlock(n_filters * 32, n_filters * 16)
        self.up2 = ImprovedUpBlock(n_filters * 16, n_filters * 8)
        self.up3 = ImprovedUpBlock(n_filters * 8, n_filters * 4)
        self.up4 = ImprovedUpBlock(n_filters * 4, n_filters * 2)
        self.up5 = ImprovedUpBlock(n_filters * 2, n_filters)

        self.smooth_conv = nn.Sequential(
            nn.Conv2d(n_filters, n_filters // 2, 3, padding=1),
            nn.BatchNorm2d(n_filters // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters // 2, n_filters // 4, 3, padding=1),
            nn.BatchNorm2d(n_filters // 4),
            nn.ReLU(inplace=True)
        )

        self.outc = nn.Conv2d(n_filters // 4, n_classes, 1)

    def forward(self, x):
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        x4, skip4 = self.down4(x3)
        x5, skip5 = self.down5(x4)

        x6, skip6 = self.bottleneck(x5)

        x = self.up1(x6, skip5)
        x = self.up2(x, skip4)
        x = self.up3(x, skip3)
        x = self.up4(x, skip2)
        x = self.up5(x, skip1)

        x = self.smooth_conv(x)
        x = self.outc(x)
        return x