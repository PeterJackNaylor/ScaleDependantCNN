from sresnet import convolution_rn_block
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SizeInvBlock(nn.Module):
    def __init__(
        self,
        inchannels,
        outchannels,
        kernel_size,
        input_size=32,
        scale_factor=3,
        gpu=False,
    ):
        super().__init__()
        # Register parameters that are trainable
        self.gpu = gpu
        self.input_size = input_size
        self.sf = scale_factor
        self.width = self.height = self.sf * self.input_size
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.randn(
                outchannels,
                inchannels,
                kernel_size[0],
                kernel_size[1],
            )
        )
        self.bias = nn.Parameter(torch.randn(outchannels))
        self.reset_parameter()

        self.upsample = nn.Upsample(
            (self.input_size * self.sf, self.input_size * self.sf)
        )
        self.bn1 = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(self.sf)

    def forward(self, x, h, w):
        x = self.upsample(x)
        bs = x.size(0)

        res = torch.empty(
            (bs, self.outchannels, self.height, self.width),
            dtype=x.dtype,
        )
        if self.gpu:
            res = res.cuda()

        for i in range(bs):
            dil_scale = (
                max(int(self.height / h[i]), 1),
                max(int(self.width / w[i]), 1),
            )
            res[i] = F.conv2d(
                x[i:i + 1],
                self.weight,
                bias=self.bias,
                stride=1,
                padding="same",
                dilation=dil_scale,
            )
        # Do a functional call so we can
        # use the same weights but different arguments
        res = self.maxpool(res)
        res = self.bn1(res)
        res = self.relu(res)
        return res

    def reset_parameter(self):
        n = self.inchannels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


class Model(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        inject_size=True,
        input_size=32,
        scale_factor=3,
        gpu=False,
    ):
        super().__init__()

        self.inplanes = 32
        self.inject_size = inject_size
        self.size_block = SizeInvBlock(
            3, self.inplanes, (3, 3), input_size, scale_factor, gpu
        )
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
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
        x = self.size_block(x, h, w)

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
    def __init__(
        self,
        block,
        layers,
        inject_size=True,
        feature_dim=64,
        input_size=32,
        scale_factor=3,
        gpu=False,
    ):
        super().__init__()

        self.inplanes = 32
        self.inject_size = inject_size
        self.size_block = SizeInvBlock(
            3, self.inplanes, (3, 3), input_size, scale_factor, gpu
        )

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
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
        x = self.size_block(x, h, w)

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
        num_classes=1000,
        inject_size=True,
        input_size=32,
        scale_factor=3,
        gpu=False,
    ):
        super().__init__()

        self.gpu = gpu
        self.inplanes = 32
        self.inject_size = inject_size
        self.size_block = SizeInvBlock(
            3, self.inplanes, (3, 3), input_size, scale_factor, gpu
        )
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
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
        x = self.size_block(x, h, w)

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
