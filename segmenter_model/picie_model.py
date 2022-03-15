import torch
import torch.nn as nn
import torch.nn.functional as F
from . import backbone_picie as backbone


class PanopticFPN(nn.Module):
    def __init__(self, args):
        super(PanopticFPN, self).__init__()
        self.backbone = backbone.__dict__[args.arch](pretrained=args.pretrain)
        if args.arch == 'vit_small':
            self.decoder = FPNDecoderViT(args)
        else:
            self.decoder = FPNDecoder(args)

    def forward(self, x, encoder_features=False, decoder_features=False):
        feats = self.backbone(x)
        dec_outs = self.decoder(feats)

        if encoder_features:
            return feats['res5'], dec_outs
        else:
            return dec_outs


class FPNDecoder(nn.Module):
    def __init__(self, args):
        super(FPNDecoder, self).__init__()
        if args.arch == 'resnet18':
            mfactor = 1
            out_dim = 128
        else:
            mfactor = 4
            out_dim = 256

        self.layer4 = nn.Conv2d(512 * mfactor // 8, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer3 = nn.Conv2d(512 * mfactor // 4, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(512 * mfactor // 2, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer1 = nn.Conv2d(512 * mfactor, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        o1 = self.layer1(x['res5'])
        o2 = self.upsample_add(o1, self.layer2(x['res4']))
        o3 = self.upsample_add(o2, self.layer3(x['res3']))
        o4 = self.upsample_add(o3, self.layer4(x['res2']))

        return o4

    def upsample_add(self, x, y):
        _, _, H, W = y.size()

        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y


class FPNDecoderViT(nn.Module):
    def __init__(self, args):
        super(FPNDecoderViT, self).__init__()
        if args.arch == 'resnet18' or args.arch == 'vit_small':
            mfactor = 1
            out_dim = 128
        else:
            mfactor = 4
            out_dim = 256

        self.upsample_rate = 4

        self.layer4 = nn.Conv2d(384, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer3 = nn.Conv2d(384, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(384, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer1 = nn.Conv2d(384, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        o1 = self.layer1(x[3])
        o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)
        o2 = self.upsample_add(o1, self.layer2(x[2]))
        o3 = self.upsample_add(o2, self.layer3(x[1]))
        o4 = self.upsample_add(o3, self.layer4(x[0]))

        return o4

    def upsample_add(self, x, y):
        return F.interpolate(y, scale_factor=self.upsample_rate, mode='bilinear', align_corners=False) + x
