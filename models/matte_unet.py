import torch
import torch.nn as nn

def conv_block(cin, cout, norm='gn', act='silu'):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False)]
    if norm == 'bn':
        layers.append(nn.BatchNorm2d(cout))
    elif norm == 'gn':
        groups = 8 if cout % 8 == 0 else max(1, cout//8)
        layers.append(nn.GroupNorm(groups, cout))
    layers.append(nn.SiLU(inplace=True) if act=='silu' else nn.ReLU(inplace=True))
    layers += [nn.Conv2d(cout, cout, 3, padding=1, bias=False)]
    if norm == 'bn':
        layers.append(nn.BatchNorm2d(cout))
    elif norm == 'gn':
        groups = 8 if cout % 8 == 0 else max(1, cout//8)
        layers.append(nn.GroupNorm(groups, cout))
    layers.append(nn.SiLU(inplace=True) if act=='silu' else nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class MatteUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, widths=(64,128,256,512,1024), norm='gn', act='silu'):
        super().__init__()
        w1,w2,w3,w4,w5 = widths
        self.e1 = conv_block(in_ch, w1, norm, act); self.p1 = nn.MaxPool2d(2)
        self.e2 = conv_block(w1, w2, norm, act);    self.p2 = nn.MaxPool2d(2)
        self.e3 = conv_block(w2, w3, norm, act);    self.p3 = nn.MaxPool2d(2)
        self.e4 = conv_block(w3, w4, norm, act);    self.p4 = nn.MaxPool2d(2)
        self.b  = conv_block(w4, w5, norm, act)
        self.u4 = nn.ConvTranspose2d(w5, w4, 2, 2); self.d4 = conv_block(w5, w4, norm, act)
        self.u3 = nn.ConvTranspose2d(w4, w3, 2, 2); self.d3 = conv_block(w4, w3, norm, act)
        self.u2 = nn.ConvTranspose2d(w3, w2, 2, 2); self.d2 = conv_block(w3, w2, norm, act)
        self.u1 = nn.ConvTranspose2d(w2, w1, 2, 2); self.d1 = conv_block(w2, w1, norm, act)
        self.out = nn.Conv2d(w1, out_ch, 1)
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        e4 = self.e4(self.p3(e3))
        b  = self.b(self.p4(e4))
        x  = self.u4(b);  x  = self.d4(torch.cat([x, e4], dim=1))
        x  = self.u3(x);  x  = self.d3(torch.cat([x, e3], dim=1))
        x  = self.u2(x);  x  = self.d2(torch.cat([x, e2], dim=1))
        x  = self.u1(x);  x  = self.d1(torch.cat([x, e1], dim=1))
        return self.out(x)
