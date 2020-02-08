import torch.nn as nn
import torch
from collections import OrderedDict
import itertools
from inspect import signature

from functools import partial
from models.twod_models.temporal_modeling import temporal_modeling_module


__all__ = ['bLVNet', 'blvnet_old']

model_urls = {
    'blresnet50': 'pretrained/ImageNet-bLResNet-50-a2-b4.pth.tar',
    'blresnet101': 'pretrained/ImageNet-bLResNet-101-a2-b4.pth.tar',
    'blresnet152': 'pretrained/ImageNet-bLResNet-152-a2-b4.pth.tar',
}

def get_frame_list(init_list, num_frames, batch_size):
    if batch_size == 0:
        return []

    flist = list()
    for i in range(batch_size):
        flist.append([k + i * num_frames for k in init_list])
    return list(itertools.chain(*flist))

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, num_frames, stride=1, downsample=None, last_relu=True,
                 temporal_module=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes // self.expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes // self.expansion)
        self.conv2 = nn.Conv2d(planes // self.expansion, planes // self.expansion, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes // self.expansion)
        self.conv3 = nn.Conv2d(planes // self.expansion, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

        self.tsm = temporal_module(duration=num_frames, channels=inplanes) \
            if temporal_module is not None else None

    def forward(self, x):
        residual = x

        if self.tsm is not None:
            x = self.tsm(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.last_relu:
            out = self.relu(out)

        return out

class bLModule(nn.Module):
    def __init__(self, block, in_channels, out_channels, blocks, alpha, beta, stride, num_frames,
                 temporal_module=None):
        super(bLModule, self).__init__()
        self.num_frames = num_frames
        self.tsm_module = temporal_module
        self.temporal_module = temporal_module

        self.relu = nn.ReLU(inplace=True)
        # only apply tsm to the BIG net
        self.big = self._make_layer(block, in_channels, out_channels, blocks - 1, 2, last_relu=False)
        self.little = self._make_layer(block, in_channels, out_channels // alpha, max(1, blocks // beta - 1))
        self.little_e = nn.Sequential(
            nn.Conv2d(out_channels // alpha, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.fusion = self._make_layer(block, out_channels, out_channels, 1, stride=stride)
        self.tsm = temporal_module(duration=self.num_frames, channels=in_channels) \
            if temporal_module is not None else None

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, last_relu=True):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes))
        downsample = None if downsample == [] else nn.Sequential(*downsample)

        layers = []
        if blocks == 1:
            layers.append(block(inplanes, planes, self.num_frames // 2, stride=stride, downsample=downsample))
        else:
            layers.append(block(inplanes, planes, self.num_frames // 2, stride, downsample))
            for i in range(1, blocks):
                layers.append(block(planes, planes, self.num_frames // 2,
                                    last_relu=last_relu if i == blocks - 1 else True))

        return nn.Sequential(*layers)

    def forward(self, x, bL_ratio, big_frame_num, big_list, little_frame_num, little_list):
        n = x.size()[0]
        if self.tsm is not None:
            x = self.tsm(x)

        big = self.big(x[big_list, ::])
        little = self.little(x[little_list, ::])
        little = self.little_e(little)
        big = torch.nn.functional.interpolate(big, little.shape[2:])

        # [-1 0 1] sum up previous, current and next frames
        bn = big_frame_num
        ln = little_frame_num

        big = big.view((-1, bn) + big.size()[1:])
        little = little.view((-1, ln) + little.size()[1:])
        big += little  # left frame

        # only do the big branch
        big = big.view((-1,) + big.size()[2:])
        big = self.relu(big)
        big = self.fusion(big)

        x = torch.zeros((n,) + big.size()[1:], device=big.device, dtype=big.dtype)
        x[range(0, n, 2), ::] = big
        x[range(1, n, 2), ::] = big

        return x

class bLVNet(nn.Module):

    def __init__(self, depth, block, layers, alpha, beta, num_frames, num_classes,
                 temporal_module=None, input_channels=3):
        self.depth = depth
        self.alpha = alpha
        self.beta = beta
        self.num_frames = num_frames
        self.temporal_module = temporal_module
        self.orig_num_frames = num_frames
        # make sure the number frames are valid
        self.bL_ratio = 2
        self.big_list = range(self.bL_ratio//2, num_frames, self.bL_ratio)
        self.little_list = list(set(range(0, num_frames)) - set(self.big_list))

        num_channels = [64, 128, 256, 512]
        self.inplanes = 64
        super(bLVNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, num_channels[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.b_conv0 = nn.Conv2d(num_channels[0], num_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_b0 = nn.BatchNorm2d(num_channels[0])
        self.l_conv0 = nn.Conv2d(num_channels[0], num_channels[0] // alpha,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_l0 = nn.BatchNorm2d(num_channels[0] // alpha)
        self.l_conv1 = nn.Conv2d(num_channels[0] // alpha, num_channels[0] //
                                 alpha, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_l1 = nn.BatchNorm2d(num_channels[0] // alpha)
        self.l_conv2 = nn.Conv2d(num_channels[0] // alpha, num_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_l2 = nn.BatchNorm2d(num_channels[0])

        self.bl_init = nn.Conv2d(num_channels[0], num_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn_bl_init = nn.BatchNorm2d(num_channels[0])
        self.tsm0 = temporal_module(duration=self.num_frames, channels=num_channels[0]) \
            if temporal_module is not None else None

        self.layer1 = bLModule(block, num_channels[0], num_channels[0] * block.expansion,
                               layers[0], alpha, beta, stride=2, num_frames=self.num_frames,
                               temporal_module=self.temporal_module)
        self.layer2 = bLModule(block, num_channels[0] * block.expansion,
                               num_channels[1] * block.expansion, layers[1], alpha, beta, stride=2, num_frames=self.num_frames,
                               temporal_module=self.temporal_module)
        self.layer3 = bLModule(block, num_channels[1] * block.expansion,
                               num_channels[2] * block.expansion, layers[2], alpha, beta, stride=1, num_frames=self.num_frames,
                               temporal_module=self.temporal_module)

        self.num_frames = self.num_frames // 2
        self.layer4 = self._make_layer(
            block, num_channels[2] * block.expansion, num_channels[3] * block.expansion, layers[3], stride=2)

        self.gappool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each block.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def mean(self, modality='rgb'):
        return [0.485, 0.456, 0.406] if modality == 'rgb' else [0.5]

    def std(self, modality='rgb'):
        return [0.229, 0.224, 0.225] if modality == 'rgb' else [np.mean([0.229, 0.224, 0.225])]

    @property
    def network_name(self):
        name = ''
        if self.temporal_module is not None:
            param = signature(self.temporal_module).parameters
            temporal_module = str(param['name']).split("=")[-1][1:-1]
            blending_frames = str(param['blending_frames']).split("=")[-1]
            blending_method = str(param['blending_method']).split("=")[-1][1:-1]
            dw_conv = True if str(param['dw_conv']).split("=")[-1] == 'True' else False
            name += "{}-b{}-{}{}-".format(temporal_module, blending_frames,
                                         blending_method,
                                         "" if dw_conv else "-allc")
        name += 'fast_blresnet-old-{}-a{}-b{}'.format(self.depth, self.alpha, self.beta)

        return name



    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = []
        if stride != 1:
            downsample.append(nn.AvgPool2d(3, stride=2, padding=1))
        if inplanes != planes:
            downsample.append(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
            downsample.append(nn.BatchNorm2d(planes))
        downsample = None if downsample == [] else nn.Sequential(*downsample)

        layers = []
        layers.append(block(inplanes, planes, self.num_frames, stride, downsample,
                            temporal_module=self.temporal_module))
        for i in range(1, blocks):
            layers.append(block(planes, planes, self.num_frames,
                                temporal_module=self.temporal_module))

        return nn.Sequential(*layers)

    def _forward_bL_layer0(self, x, bL_ratio, big_frame_num, big_list, little_frame_num, little_list):
        n = x.size()[0]
        if self.tsm0 is not None:
            x = self.tsm0(x)

        bx = self.b_conv0(x[big_list, ::])
        bx = self.bn_b0(bx)

        lx = self.l_conv0(x[little_list, ::])
        lx = self.bn_l0(lx)
        lx = self.relu(lx)
        lx = self.l_conv1(lx)
        lx = self.bn_l1(lx)
        lx = self.relu(lx)
        lx = self.l_conv2(lx)
        lx = self.bn_l2(lx)

        bn = big_frame_num
        ln = little_frame_num
        bx = bx.view((-1, bn) + bx.size()[1:])
        lx = lx.view((-1, ln) + lx.size()[1:])
        bx += lx   # left frame
        bx = bx.view((-1,) + bx.size()[2:])

        bx = self.relu(bx)
        bx = self.bl_init(bx)
        bx = self.bn_bl_init(bx)
        bx = self.relu(bx)

        x = torch.zeros((n,) + bx.size()[1:], device=bx.device, dtype=bx.dtype)
        x[range(0, n, 2), ::] = bx
        x[range(1, n, 2), ::] = bx

        return x

    def forward(self, x):

        batch_size, c_t, h, w = x.shape
        x = x.view(batch_size * self.orig_num_frames, c_t // self.orig_num_frames, h, w)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        big_list = get_frame_list(self.big_list, self.orig_num_frames, batch_size)
        little_list = get_frame_list(self.little_list, self.orig_num_frames, batch_size)

        x = self._forward_bL_layer0(x, self.bL_ratio, len(self.big_list), big_list, len(self.little_list), little_list)
        x = self.layer1(x, self.bL_ratio, len(self.big_list), big_list, len(self.little_list), little_list)
        x = self.layer2(x, self.bL_ratio, len(self.big_list), big_list, len(self.little_list), little_list)
        x = self.layer3(x, self.bL_ratio, len(self.big_list), big_list, len(self.little_list), little_list)

        x = self.layer4(x[big_list, ::])

        x = self.gappool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        n_t, c = x.shape
        out = x.view(batch_size, -1, c)

        # average the prediction from all frames
        out = torch.mean(out, dim=1)
        return out

def blvnet_old(depth, alpha, beta, groups,
               temporal_module_name, num_classes,
               pretrained=True, blending_frames=3, blending_method='sum',
               input_channels=3, **kwargs):
    layers = {
        50: [3, 4, 6, 3],
        101: [4, 8, 18, 3],
        152: [5, 12, 30, 3]
    }[depth]

    temporal_module = partial(temporal_modeling_module, name=temporal_module_name,
                              dw_conv=True,
                              blending_frames=blending_frames,
                              blending_method=blending_method) if temporal_module_name is not None \
        else None

    model = bLVNet(depth, Bottleneck, layers, alpha, beta, groups, num_classes,
                   temporal_module=temporal_module, input_channels=input_channels)

    if pretrained:
        checkpoint = torch.load(model_urls['blresnet{}'.format(depth)], map_location='cpu')
        # fixed parameter names in order to load the weights correctly
        state_d = OrderedDict()
        if input_channels == 3:
            for key, value in checkpoint['state_dict'].items():
                state_d[key.replace('module.', '')] = value
        else:  # TODO: assume it is flow.
            for key, value in checkpoint['state_dict'].items():
                new_key = key.replace('module.', '')
                if "module.conv1.weight" in key:
                    o_c, in_c, k_h, k_w = value.shape
                else:
                    o_c, in_c, k_h, k_w = 0, 0, 0, 0
                if k_h == 7 and k_w == 7:
                    # average the weights and expand to all channels
                    new_shape = (o_c, input_channels, k_h, k_w)
                    new_value = value.mean(dim=1, keepdim=True).expand(new_shape).contiguous()
                else:
                    new_value = value
                state_d[new_key] = new_value
        state_d.pop('fc.weight', None)
        state_d.pop('fc.bias', None)
        model.load_state_dict(state_d, strict=False)
    return model


if __name__ == "__main__":
    model = blvnet_old(101, 2, 4, 16, 'TAM', 174, False)
    print(model.mean())
    print(model.network_name)