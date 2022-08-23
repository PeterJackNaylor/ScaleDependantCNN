import torch
import torch.nn as nn
import torch.nn.functional as F


def convolution_rn_block(block, inplanes, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride, bias=False),
            nn.BatchNorm2d(planes),
        )
    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes))
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Model(nn.Module):
    def __init__(self, block, layers, kernel_size=3, num_classes=1000, inject_size=False):
        super().__init__()

        self.inplanes = 32
        self.inject_size = inject_size

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=kernel_size, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = convolution_rn_block(block, 32, 32, layers[0])
        self.layer2 = convolution_rn_block(block, 32, 64, layers[1], stride=2)
        self.layer3 = convolution_rn_block(block, 64, 128, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.fc_inject = nn.Linear(130, 128)
        self.fc_inject_bn = nn.BatchNorm1d(128)
        self.fc_inject_relu = nn.ReLU(inplace=True)

    def forward(self, x, h=0, w=0, return_embedding=False):
        x = self.conv1(x)  # 32x32
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)  # 1x1
        # remove 1 X 1 grid and make vector of tensor shape
        emb = torch.flatten(x, 1)
        if self.inject_size:
            emb = torch.cat((emb, h, w), axis=1)
            emb = self.fc_inject(emb)
            emb = self.fc_inject_bn(emb)
            emb = self.fc_inject_relu(emb)
        x = self.fc(emb)
        if return_embedding:
            return x, emb
        return x


class Model_ssl(nn.Module):
    def __init__(self, block, layers, kernel_size=3, inject_size=False, feature_dim=64):
        super().__init__()

        self.inplanes = 32
        self.inject_size = inject_size

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=kernel_size, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = convolution_rn_block(block, 32, 32, layers[0])
        self.layer2 = convolution_rn_block(block, 32, 64, layers[1], stride=2)
        self.layer3 = convolution_rn_block(block, 64, 128, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_inject = nn.Linear(130, 128)
        self.fc_inject_bn = nn.BatchNorm1d(128)
        self.fc_inject_relu = nn.ReLU(inplace=True)

        self.g = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feature_dim, bias=True),
        )

    def forward(self, x, h=0, w=0):
        x = self.conv1(x)  # 32x32
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)  # 1x1
        # remove 1 X 1 grid and make vector of tensor shape
        emb = torch.flatten(x, 1)
        if self.inject_size:
            emb = torch.cat((emb, h, w), axis=1)
            emb = self.fc_inject(emb)
            emb = self.fc_inject_bn(emb)
            emb = self.fc_inject_relu(emb)

        out = self.g(emb)

        return F.normalize(emb, dim=-1), F.normalize(out, dim=-1)


class Model_backbone(nn.Module):
    def __init__(
        self,
        block,
        layers,
        kernel_size=3,
        num_classes=1000,
        inject_size=False,
        gpu=True,
    ):
        super().__init__()
        self.gpu = gpu
        self.inplanes = 32
        self.inject_size = inject_size

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=kernel_size, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = convolution_rn_block(block, 32, 32, layers[0])
        self.layer2 = convolution_rn_block(block, 32, 64, layers[1], stride=2)
        self.layer3 = convolution_rn_block(block, 64, 128, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.fc_inject = nn.Linear(130, 128)
        self.fc_inject_bn = nn.BatchNorm1d(128)
        self.fc_inject_relu = nn.ReLU(inplace=True)

    def forward(self, x, h=0, w=0):
        x = self.conv1(x)  # 32x32
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)  # 1x1
        # remove 1 X 1 grid and make vector of tensor shape
        emb = torch.flatten(x, 1)
        if self.inject_size:
            if self.gpu:
                h = h.cuda()
                w = w.cuda()
            emb = torch.cat((emb, h, w), axis=1)
            emb = self.fc_inject(emb)
            emb = self.fc_inject_bn(emb)
            emb = self.fc_inject_relu(emb)
        return emb
