from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

# from timm.models.layers import trunc_normal_

from segmenter_model.utils import padding, unpadding


class Segmenter(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im, decoder_features=False, no_upsample=False, encoder_features=False, no_rearrange=False,
                cls_only=False, encoder_only=False):
        H_ori, W_ori = im.size(2), im.size(3)
        if not no_upsample:
            im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im, return_features=True)  # self.patch_size times smaller than im

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled

        if cls_only:
            return x[:, 0]
        x = x[:, num_extra_tokens:]

        if encoder_features:
            enc_fts = x.clone()
            if not no_rearrange:
                GS = H // self.patch_size
                enc_fts = rearrange(enc_fts, "b (h w) c -> b c h w", h=GS)
            if encoder_only:
                return enc_fts

        if decoder_features:
            output = self.decoder(x, (H, W), features_only=True, no_rearrange=no_rearrange)
            if no_rearrange:
                if encoder_features:
                    output = (enc_fts, output)
                return output
        else:
            output = self.decoder(x, (H, W))  # shape (BS, NCLS, H/self.patch_size, W/self.patch_size)

        if not no_upsample:
            output = F.interpolate(output, size=(H, W), mode="bilinear")  # upsample self.patch_size times
            output = unpadding(output, (H_ori, W_ori))

        if encoder_features:
            output = (enc_fts, output)
        return output

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
