import timm
import torchvision.models as models
from torch.nn import init
import torch.nn.functional as F
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x

class Resnet(nn.Module):
    def __init__(self, model, num_classes):
        super(Resnet, self).__init__()

        # 提取 ResNet 的特征部分
        self.features = nn.Sequential(
            model.conv1,      # [batch_size, 64, H/2, W/2]
            model.bn1,
            model.relu,
            model.maxpool,    # [batch_size, 64, H/4, W/4]
            model.layer1,     # [batch_size, 256, H/4, W/4]
            model.layer2,     # [batch_size, 512, H/8, W/8]
            model.layer3,     # [batch_size, 1024, H/16, W/16]
            model.layer4      # [batch_size, 2048, H/32, W/32]
        )
        num_features = model.layer4[1].conv1.in_channels
        # 全局平均池化，将特征图的空间维度压缩为 1x1
        self.attention = SEBlock(num_features)  # 使用Squeeze-and-Excitation块作为颜色注意力
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # [batch_size, 2048, 1, 1]

        # 分类层
        self.classifier = nn.Sequential(
            nn.Flatten(),                           # [batch_size, 2048]
            nn.Linear(num_features, 512),                   # 可以根据需要调整隐藏层大小
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),                        # 防止过拟合
            nn.Linear(512, num_classes)             # 输出层，5 类分类
        )
        self.att = SpatialGroupEnhance(32)
    def forward(self, inputs):
        x = self.features(inputs)
        # x = self.att(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x




def resnet(num_classes=4, pretrained=False):
    # 加载预训练的 ResNet50 模型
    model = models.resnet101(pretrained=pretrained)

    # 创建自定义的 Resnet 模型，添加分类层
    custom_resnet = Resnet(model, num_classes)

    return custom_resnet