import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, inplanes, planes, size, stride=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(inplanes,planes,size,padding=size//2,stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        return self.bn(torch.relu(self.conv(x)))

    def get_weights_l2_norm(self):
        return torch.linalg.norm(self.conv.weight)


class ConvNet(nn.Module):
    def __init__(self, planes, size = 3):
        super(ConvNet,self).__init__()
        self.c1 = Conv(3,planes,size)
        self.p1 = Conv(planes,planes*2,size,2)
        self.c2 = Conv(planes*2,planes*2,size)
        self.p2 = Conv(planes*2,planes*4,size+2,4)
        self.c3 = Conv(planes*4,planes*4,size)
        self.p3 = Conv(planes*4,planes*8,size+2,4)

        self.classifier = nn.Conv2d(planes*8,5,1)

    def forward(self, x):
        x = self.p1(self.c1(x))
        x = self.p2(self.c2(x))
        x = self.p3(self.c3(x))
        return torch.squeeze(self.classifier(x))

    def get_weights_l2_norm(self):
        c1_w = self.c1.get_weights_l2_norm()
        c2_w = self.c2.get_weights_l2_norm()
        c3_w = self.c3.get_weights_l2_norm()
        p1_w = self.p1.get_weights_l2_norm()
        p2_w = self.p2.get_weights_l2_norm()
        p3_w = self.p3.get_weights_l2_norm()
        cls_w = torch.linalg.norm(self.classifier.weight)
        return sum([c1_w, p1_w, c2_w, p2_w, c3_w, p3_w, cls_w])/7.0
