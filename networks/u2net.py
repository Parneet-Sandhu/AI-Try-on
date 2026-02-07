import torch
import torch.nn as nn
import torch.nn.functional as F


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


class RSU(nn.Module):
    def __init__(self, name, height, in_ch=3, mid_ch=12, out_ch=3, dirate=1):
        super(RSU, self).__init__()
        self.name = name
        self.height = height
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=dirate)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=dirate)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=dirate)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=dirate)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=dirate)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=dirate)

        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=dirate)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=dirate)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=dirate)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=dirate)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx5up = F.interpolate(hx5, size=hx4.size()[2:], mode='bilinear', align_corners=False)
        hx4d = self.rebnconv4d(torch.cat((hx5up, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=hx3.size()[2:], mode='bilinear', align_corners=False)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=hx2.size()[2:], mode='bilinear', align_corners=False)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=hx1.size()[2:], mode='bilinear', align_corners=False)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()

        self.stage1 = RSU("stage1", 7, in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU("stage2", 6, 64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU("stage3", 5, 128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU("stage4", 4, 256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU("stage5", 3, 512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU("stage6", 2, 512, 256, 512)

        # decoder
        self.stage5d = RSU("stage5d", 3, 1024, 256, 512)
        self.stage4d = RSU("stage4d", 4, 1024, 128, 256)
        self.stage3d = RSU("stage3d", 5, 512, 64, 128)
        self.stage2d = RSU("stage2d", 6, 256, 32, 64)
        self.stage1d = RSU("stage1d", 7, 128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        h1 = self.stage1(hx)
        hx = self.pool12(h1)

        # stage 2
        h2 = self.stage2(hx)
        hx = self.pool23(h2)

        # stage 3
        h3 = self.stage3(hx)
        hx = self.pool34(h3)

        # stage 4
        h4 = self.stage4(hx)
        hx = self.pool45(h4)

        # stage 5
        h5 = self.stage5(hx)
        hx = self.pool56(h5)

        # stage 6
        h6 = self.stage6(hx)

        # decoder
        hx = F.interpolate(h6, size=h5.size()[2:], mode='bilinear', align_corners=False)
        hx = torch.cat((hx, h5), 1)
        h5d = self.stage5d(hx)

        hx = F.interpolate(h5d, size=h4.size()[2:], mode='bilinear', align_corners=False)
        hx = torch.cat((hx, h4), 1)
        h4d = self.stage4d(hx)

        hx = F.interpolate(h4d, size=h3.size()[2:], mode='bilinear', align_corners=False)
        hx = torch.cat((hx, h3), 1)
        h3d = self.stage3d(hx)

        hx = F.interpolate(h3d, size=h2.size()[2:], mode='bilinear', align_corners=False)
        hx = torch.cat((hx, h2), 1)
        h2d = self.stage2d(hx)

        hx = F.interpolate(h2d, size=h1.size()[2:], mode='bilinear', align_corners=False)
        hx = torch.cat((hx, h1), 1)
        h1d = self.stage1d(hx)

        # side outputs
        d1 = self.side1(h1d)
        d2 = self.side2(h2d)
        d2 = F.interpolate(d2, size=d1.size()[2:], mode='bilinear', align_corners=False)
        d3 = self.side3(h3d)
        d3 = F.interpolate(d3, size=d1.size()[2:], mode='bilinear', align_corners=False)
        d4 = self.side4(h4d)
        d4 = F.interpolate(d4, size=d1.size()[2:], mode='bilinear', align_corners=False)
        d5 = self.side5(h5d)
        d5 = F.interpolate(d5, size=d1.size()[2:], mode='bilinear', align_corners=False)
        d6 = self.side6(h6)
        d6 = F.interpolate(d6, size=d1.size()[2:], mode='bilinear', align_corners=False)

        dout = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return dout
