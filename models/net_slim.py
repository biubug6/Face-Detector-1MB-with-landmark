import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def depth_conv2d(inp, oup, kernel=1, stride=1, pad=0):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size = kernel, stride = stride, padding=pad, groups=inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, kernel_size=1)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class Slim(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(Slim, self).__init__()
        self.phase = phase
        self.num_classes = 2

        self.conv1 = conv_bn(3, 16, 2)
        self.conv2 = conv_dw(16, 32, 1)
        self.conv3 = conv_dw(32, 32, 2)
        self.conv4 = conv_dw(32, 32, 1)
        self.conv5 = conv_dw(32, 64, 2)
        self.conv6 = conv_dw(64, 64, 1)
        self.conv7 = conv_dw(64, 64, 1)
        self.conv8 = conv_dw(64, 64, 1)

        self.conv9 = conv_dw(64, 128, 2)
        self.conv10 = conv_dw(128, 128, 1)
        self.conv11 = conv_dw(128, 128, 1)

        self.conv12 = conv_dw(128, 256, 2)
        self.conv13 = conv_dw(256, 256, 1)

        self.conv14 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True),
            depth_conv2d(64, 256, kernel=3, stride=2, pad=1),
            nn.ReLU(inplace=True)
        )
        self.loc, self.conf, self.landm = self.multibox(self.num_classes);

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []
        landm_layers = []
        loc_layers += [depth_conv2d(64, 3 * 4, kernel=3, pad=1)]
        conf_layers += [depth_conv2d(64, 3 * num_classes, kernel=3, pad=1)]
        landm_layers += [depth_conv2d(64, 3 * 10, kernel=3, pad=1)]

        loc_layers += [depth_conv2d(128, 2 * 4, kernel=3, pad=1)]
        conf_layers += [depth_conv2d(128, 2 * num_classes, kernel=3, pad=1)]
        landm_layers += [depth_conv2d(128, 2 * 10, kernel=3, pad=1)]

        loc_layers += [depth_conv2d(256, 2 * 4, kernel=3, pad=1)]
        conf_layers += [depth_conv2d(256, 2 * num_classes, kernel=3, pad=1)]
        landm_layers += [depth_conv2d(256, 2 * 10, kernel=3, pad=1)]

        loc_layers += [nn.Conv2d(256, 3 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, 3 * num_classes, kernel_size=3, padding=1)]
        landm_layers += [nn.Conv2d(256, 3 * 10, kernel_size=3, padding=1)]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers), nn.Sequential(*landm_layers)


    def forward(self,inputs):
        detections = list()
        loc = list()
        conf = list()
        landm = list()

        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        detections.append(x8)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        detections.append(x11)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        detections.append(x13)

        x14= self.conv14(x13)
        detections.append(x14)

        for (x, l, c, lam) in zip(detections, self.loc, self.conf, self.landm):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            landm.append(lam(x).permute(0, 2, 3, 1).contiguous())

        bbox_regressions = torch.cat([o.view(o.size(0), -1, 4) for o in loc], 1)
        classifications = torch.cat([o.view(o.size(0), -1, 2) for o in conf], 1)
        ldm_regressions = torch.cat([o.view(o.size(0), -1, 10) for o in landm], 1)



        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output
