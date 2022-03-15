import math
# import segm.utils.torch as ptu
# from segm.engine import seg2rgb
from collections import namedtuple

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

import torch

CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                 'has_instances', 'ignore_in_eval', 'color'])

classes = [
    CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

cityscapes_id_to_trainID = {cls.id: cls.train_id for cls in classes}
cityscapes_trainID_to_testID = {cls.train_id: cls.id for cls in classes}
cityscapes_trainID_to_color = {cls.train_id: cls.color for cls in classes}
cityscapes_trainID_to_name = {cls.train_id: cls.name for cls in classes}
cityscapes_trainID_to_color[255] = (0, 0, 0)
cityscapes_trainID_to_name = {cls.train_id: cls.name for cls in classes}
cityscapes_trainID_to_name[255] = 'ignore'
cityscapes_trainID_to_name[19] = 'ignore'


def map2cs(seg):
    while len(seg.shape) > 2:
        seg = seg[0]
    colors = cityscapes_trainID_to_color
    # assert False, 'set ignore_idx color to black, make sure that it is not in colors'
    rgb = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for l in np.unique(seg):
        rgb[seg == l, :] = colors[l]
    return rgb


def get_colors(num_colors):
    from PIL import ImageColor
    import matplotlib
    hex_colors = [
        # "#000000", # keep the black reserved
        "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
        "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
        "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58",
    ]
    hex_colors_mlib = list(matplotlib.colors.cnames.values())
    for hcm in hex_colors_mlib:
        if hcm not in hex_colors:
            hex_colors.append(hcm)
    colors = [ImageColor.getrgb(hex) for hex in hex_colors]
    return colors[:num_colors]


def colorize_one(seg, ignore=None, colors=None, ncolors=32):
    unq = np.unique(seg)
    if ncolors is not None:
        ncolors = max(ncolors, max(unq))
    else:
        ncolors = max(unq)
    colors = get_colors(ncolors) if colors is None else colors
    h, w = seg.shape
    c = 3
    rgb = np.zeros((h, w, c), dtype=np.uint8)
    for l in unq:
        if ignore is not None and l == ignore:
            continue
        try:
            rgb[seg == l, :] = colors[l]
        except:
            raise Exception(l)
    return rgb


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def resize_pos_embed(posemb, grid_old_shape, grid_new_shape, num_extra_tokens):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb_tok, posemb_grid = (
        posemb[:, :num_extra_tokens],
        posemb[0, num_extra_tokens:],
    )
    if grid_old_shape is None:
        gs_old_h = int(math.sqrt(len(posemb_grid)))
        gs_old_w = gs_old_h
    else:
        gs_old_h, gs_old_w = grid_old_shape

    gs_h, gs_w = grid_new_shape
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    num_extra_tokens = 1 + ("dist_token" in state_dict.keys())
    patch_size = model.patch_size
    image_size = model.patch_embed.image_size
    for k, v in state_dict.items():
        if k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v,
                None,
                (image_size[0] // patch_size, image_size[1] // patch_size),
                num_extra_tokens,
            )
        out_dict[k] = v
    return out_dict


def padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    H, W = im.size(2), im.size(3)
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        im_padded = F.pad(im, (0, pad_w, 0, pad_h), value=fill_value)
    return im_padded


def unpadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.size(2), y.size(3)
    # crop predictions on extra pixels coming from padding
    extra_h = H_pad - H
    extra_w = W_pad - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y


def resize(im, smaller_size):
    h, w = im.shape[2:]
    if h < w:
        ratio = w / h
        h_res, w_res = smaller_size, ratio * smaller_size
    else:
        ratio = h / w
        h_res, w_res = ratio * smaller_size, smaller_size
    if min(h, w) < smaller_size:
        im_res = F.interpolate(im, (int(h_res), int(w_res)), mode="bilinear")
    else:
        im_res = im
    return im_res


def sliding_window(im, flip, window_size, window_stride, channels_first=True):
    if channels_first:
        B, C, H, W = im.shape
    else:
        B, H, W, C = im.shape
    ws = window_size

    windows = {"crop": [], "anchors": []}
    h_anchors = torch.arange(0, H, window_stride)
    w_anchors = torch.arange(0, W, window_stride)
    h_anchors = [h.item() for h in h_anchors if h < H - ws] + [H - ws]
    w_anchors = [w.item() for w in w_anchors if w < W - ws] + [W - ws]
    for ha in h_anchors:
        for wa in w_anchors:
            if channels_first:
                window = im[:, :, ha: ha + ws, wa: wa + ws]
            else:
                window = im[:, ha: ha + ws, wa: wa + ws]
            windows["crop"].append(window)
            windows["anchors"].append((ha, wa))
    windows["flip"] = flip
    windows["shape"] = (H, W)
    return windows


def merge_windows(windows, window_size, ori_shape, no_softmax=False, no_upsample=False, patch_size=None):
    ws = window_size
    im_windows = windows["seg_maps"]
    anchors = windows["anchors"]
    C = im_windows[0].shape[0]
    H, W = windows["shape"]
    flip = windows["flip"]

    if no_upsample:
        H, W = H // patch_size, W // patch_size

    logit = torch.zeros((C, H, W), device=im_windows.device)
    count = torch.zeros((1, H, W), device=im_windows.device)
    for window, (ha, wa) in zip(im_windows, anchors):
        if no_upsample:
            ha = ha // patch_size
            wa = wa // patch_size
        logit[:, ha: ha + ws, wa: wa + ws] += window
        count[:, ha: ha + ws, wa: wa + ws] += 1
    logit /= count
    # print('Interpolate {} -> {}'.format(logit.shape, ori_shape))
    if not no_upsample:
        logit = F.interpolate(
            logit.unsqueeze(0),
            ori_shape,
            mode="bilinear",
        )[0]
    if flip:
        logit = torch.flip(logit, (2,))
    if not no_softmax:
        # print('Softmax in merge_windows')
        result = F.softmax(logit, 0)
    else:
        # print('No softmax in merge_windows')
        result = logit
    return result


def debug_windows(windows, debug_file):
    pass


def inference_picie(
        model,
        classifier,
        metric_test,
        ims,
        ori_shape,
        window_size,
        window_stride,
        batch_size,
        decoder_features=False,
        no_upsample=False,
        debug_file=None,
        im_rgb=None,
        channel_first=False
):
    try:
        C = model.n_cls
    except:
        C = classifier.module.bias.shape[0]

    # seg_maps = []

    # for im, im_metas in zip(ims, ims_metas):
    for im in ims:
        im = im.to('cuda')
        if len(im.shape) == 3:
            im = im.unsqueeze(0)
        flip = False  # im_metas["flip"]
        windows = sliding_window(im, flip, window_size, window_stride)
        crops = torch.stack(windows.pop("crop"))[:, 0]
        num_crops = len(crops)

        WB = batch_size if batch_size > 0 else num_crops
        if no_upsample:
            window_size = window_size // model.patch_size
        seg_maps = torch.zeros((num_crops, C, window_size, window_size), device=im.device)
        with torch.no_grad():
            for i in range(0, num_crops, WB):
                # try:
                feats = model.forward(crops[i: i + WB])
                if metric_test == 'cosine':
                    feats = F.normalize(feats, dim=1, p=2)
                probs = classifier(feats)
                probs = F.interpolate(probs, crops[i: i + WB].shape[-2:], mode='bilinear', align_corners=False)
                seg_maps[i: i + WB] = probs
        windows["seg_maps"] = seg_maps

        im_seg_map = merge_windows(windows, window_size, ori_shape, no_softmax=decoder_features,
                                   no_upsample=no_upsample, patch_size=None)

        seg_map = im_seg_map
        if no_upsample and not decoder_features:
            pass
        else:
            seg_map = F.interpolate(
                seg_map.unsqueeze(0),
                ori_shape,
                mode="bilinear",
            )

    return seg_map


def inference(
        model,
        ims,
        ori_shape,
        window_size,
        window_stride,
        batch_size,
        decoder_features=False,
        encoder_features=False,
        no_upsample=False,
):
    C = model.n_cls
    patch_size = model.patch_size

    # seg_maps = []

    # for im, im_metas in zip(ims, ims_metas):
    for im in ims:
        # im = im.to('cuda')
        if len(im.shape) == 3:
            im = im.unsqueeze(0)
        # im = resize(im, window_size)
        flip = False  # im_metas["flip"]
        # print(im)
        windows = sliding_window(im, flip, window_size, window_stride)
        # print(windows)
        crops = torch.stack(windows.pop("crop"))[:, 0]
        num_crops = len(crops)

        WB = batch_size if batch_size > 0 else num_crops
        if no_upsample:
            window_size = window_size // model.patch_size
            # print('Change variable window_size to {}'.format(window_size))
        seg_maps = torch.zeros((num_crops, C, window_size, window_size), device=im.device)
        # print('Allocated segm_maps:  {}, device: {}'.format(seg_maps.shape, seg_maps.device))
        with torch.no_grad():
            for i in range(0, num_crops, WB):
                # try:
                print('Forward crop {}'.format(crops[i: i + WB].shape))
                seg_maps[i: i + WB] = model.forward(crops[i: i + WB], decoder_features=decoder_features,
                                                    encoder_features=encoder_features,
                                                    no_upsample=no_upsample)
                # except:
                #     print('Input of shape: {}'.format(crops[i:i + WB].shape))
                #     assert False, "End after error."
                # torch.cuda.empty_cache()
        windows["seg_maps"] = seg_maps

        im_seg_map = merge_windows(windows, window_size, ori_shape, no_softmax=decoder_features,
                                   no_upsample=no_upsample, patch_size=model.patch_size)

        seg_map = im_seg_map
        if no_upsample and not decoder_features:
            pass
        else:
            seg_map = F.interpolate(
                seg_map.unsqueeze(0),
                ori_shape,
                mode="bilinear",
            )
        # seg_maps.append(seg_map)

        # print('Done one inference.')
    # seg_maps = torch.cat(seg_maps, dim=0)
    return seg_map


def inference_features(
        model,
        ims,
        ori_shape,
        window_size,
        window_stride,
        batch_size,
        decoder_features=False,
        encoder_features=False,
        save2cpu=False,
        no_upsample=True,
        encoder_only=False
):
    C = model.n_cls if decoder_features else model.encoder.d_model
    patch_size = model.patch_size

    # seg_maps = []

    # for im, im_metas in zip(ims, ims_metas):
    for im in ims:
        im = im.to('cuda')
        if len(im.shape) == 3:
            im = im.unsqueeze(0)
        # im = resize(im, window_size)
        flip = False  # im_metas["flip"]
        # print(im)
        windows = sliding_window(im, flip, window_size, window_stride)
        # print(windows)
        crops = torch.stack(windows.pop("crop"))[:, 0]
        num_crops = len(crops)

        WB = batch_size if batch_size > 0 else num_crops
        if no_upsample:
            window_size = window_size // model.patch_size
            # print('Change variable window_size to {}'.format(window_size))
        enc_maps = torch.zeros((num_crops, C, window_size, window_size), device=im.device)
        if decoder_features:
            dec_maps = torch.zeros((num_crops, C, window_size, window_size), device=im.device)
        # print('Allocated segm_maps:  {}, device: {}'.format(seg_maps.shape, seg_maps.device))
        with torch.no_grad():
            for i in range(0, num_crops, WB):
                enc_fts = model.forward(crops[i: i + WB], decoder_features=decoder_features,
                                        encoder_features=True,
                                        no_upsample=no_upsample, encoder_only=encoder_only)
                if decoder_features:
                    enc_fts, dec_fts = enc_fts
                    dec_maps[i: i + WB] = dec_fts
                elif isinstance(enc_fts, tuple):
                    enc_fts = enc_fts[0]
                enc_maps[i: i + WB] = enc_fts

        windows["seg_maps"] = enc_maps
        im_enc_map = merge_windows(windows, window_size, ori_shape, no_softmax=decoder_features,
                                   no_upsample=no_upsample, patch_size=model.patch_size)

        if decoder_features:
            windows["seg_maps"] = dec_maps
            im_dec_map = merge_windows(windows, window_size, ori_shape, no_softmax=decoder_features,
                                       no_upsample=no_upsample, patch_size=model.patch_size)

        if no_upsample:
            pass
        else:
            im_enc_map = F.interpolate(
                im_enc_map.unsqueeze(0),
                ori_shape,
                mode="bilinear",
            )
            if decoder_features:
                im_dec_map = F.interpolate(
                    im_dec_map.unsqueeze(0),
                    ori_shape,
                    mode="bilinear",
                )

    im_enc_map = im_enc_map.cpu().numpy()
    if decoder_features:
        im_dec_map = im_dec_map.cpu().numpy()
        return im_enc_map, im_dec_map

    return im_enc_map


def inference_conv(
        model,
        ims,
        ims_metas,
        ori_shape
):
    assert len(ims) == 1
    for im, im_metas in zip(ims, ims_metas):
        im = im.to(ptu.device)
        if len(im.shape) < 4:
            im = im.unsqueeze(0)
        logits = model(im)
        if ori_shape[:2] != logits.shape[-2:]:
            # resize
            logits = F.interpolate(
                logits,
                ori_shape[-2:],
                mode="bilinear",
            )
        # 3) applies softmax
        result = F.softmax(logits.squeeze(), 0)
    # print(result.shape)
    return result


def num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    if not type(n_params) == int:
        n_params = n_params.item()
    return n_params
