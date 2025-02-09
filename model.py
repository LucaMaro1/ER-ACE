import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from fvcore.nn import FlopCountAnalysis as FCA


class distLinear(nn.Module):
    def __init__(self, indim, outdim, weight=None):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        if weight is not None:
            self.L.weight.data = Variable(weight)

        self.scale_factor = 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)

        L_norm = torch.norm(self.L.weight, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
        cos_dist = torch.mm(x_normalized, self.L.weight.div(L_norm + 0.00001).transpose(0,1))

        scores = self.scale_factor * (cos_dist)

        return scores


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                                        nn.BatchNorm2d(self.expansion * planes))
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.activation(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, input_size, dist_linear=False):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.input_size = input_size

        self.conv1 = conv3x3(input_size[0], nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        last_hid = nf * 8 * block.expansion
        last_hid = last_hid * (self.input_size[-1] // 2 // 2 // 2 // 4) ** 2

        if dist_linear:
            self.linear = distLinear(last_hid, num_classes)
        else:
            self.linear = nn.Linear(last_hid, num_classes)

        self.activation = nn.ReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def return_hidden(self, x):
        bsz = x.size(0)
        assert x.ndim == 4
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.return_hidden(x)
        out = self.linear(out)
        return out

    def one_sample_flop(self, device):
        if not hasattr(self, '_train_cost'):
            input = torch.FloatTensor(size=(1,) + self.input_size).to(device)
            flops = FCA(self, input)
            self._train_cost = flops.total() / 1e6 # MegaFlops

        return self._train_cost

def ResNet18(nclasses, nf=20, input_size=(3, 32, 32), *args, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, input_size, *args, **kwargs)