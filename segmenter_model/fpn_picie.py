# taken from https://raw.githubusercontent.com/janghyuncho/PiCIE/1d7b034f57e98670b0d6a244b2eea11fa0dde73e/modules/fpn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import backbone_picie as backbone


class PanopticFPN(nn.Module):
    def __init__(self, arch, pretrain, n_cls):
        super(PanopticFPN, self).__init__()
        self.n_cls = n_cls
        self.backbone = backbone.__dict__[arch](pretrained=pretrain)
        self.decoder = FPNDecoder(arch, n_cls)

    def forward(self, x, encoder_features=False, decoder_features=False):
        feats = self.backbone(x)
        if decoder_features:
            dec, outs = self.decoder(feats, get_features=decoder_features)
        else:
            outs = self.decoder(feats)

        if encoder_features:
            if decoder_features:
                return feats['res5'], dec, outs
            else:
                return feats['res5'], outs
        else:
            return outs


class FPNDecoder(nn.Module):
    def __init__(self, arch, n_cls):
        super(FPNDecoder, self).__init__()
        self.n_cls = n_cls
        if arch == 'resnet18':
            mfactor = 1
            out_dim = 128
        else:
            mfactor = 4
            out_dim = 256

        self.layer4 = nn.Conv2d(512 * mfactor // 8, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer3 = nn.Conv2d(512 * mfactor // 4, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(512 * mfactor // 2, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer1 = nn.Conv2d(512 * mfactor, out_dim, kernel_size=1, stride=1, padding=0)

        self.pred = nn.Conv2d(out_dim, self.n_cls, 1, 1)

    def forward(self, x, get_features=False):
        o1 = self.layer1(x['res5'])
        o2 = self.upsample_add(o1, self.layer2(x['res4']))
        o3 = self.upsample_add(o2, self.layer3(x['res3']))
        o4 = self.upsample_add(o3, self.layer4(x['res2']))

        pred = self.pred(o4)

        if get_features:
            return o4, pred
        else:
            return pred

    def upsample_add(self, x, y):
        _, _, H, W = y.size()

        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y
