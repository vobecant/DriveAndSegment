import os

import click
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from segmenter_model import utils
from segmenter_model.factory import create_segmenter
from segmenter_model.fpn_picie import PanopticFPN
from segmenter_model.utils import colorize_one, map2cs

# WEIGHTS = './weights/segmenter_waymo.pth
WEIGHTS = './weights/segmenter_nusc.pth'


def blend_images(bg, fg, alpha=0.3):
    fg = fg.convert('RGBA')
    bg = bg.convert('RGBA')
    blended = Image.blend(bg, fg, alpha=alpha).convert('RGB')

    return blended


def download_weights():
    import urllib.request
    if not os.path.exists(WEIGHTS):
        d, _ = os.path.split(WEIGHTS)
        if not os.path.isdir(d):
            os.makedirs(d)
        url_weights = 'https://data.ciirc.cvut.cz/public/projects/2022DriveAndSegment/segmenter_nusc.pth'
        urllib.request.urlretrieve(url_weights, WEIGHTS)
        for cuda in [True, False]:
            variant_path = '{}_variant{}.yml'.format(WEIGHTS, '_gpu' if cuda else '')
            url_variant = 'https://data.ciirc.cvut.cz/public/projects/2022DriveAndSegment/segmenter_nusc.pth_variant{}.yml'.format(
                '_gpu' if cuda else '')
            urllib.request.urlretrieve(url_variant, variant_path)


def segment_segmenter(image, model, window_size, window_stride, encoder_features=False, decoder_features=False,
                      no_upsample=False, batch_size=1):
    seg_pred = utils.inference(
        model,
        image,
        image.shape[-2:],
        window_size,
        window_stride,
        batch_size=batch_size,
        no_upsample=no_upsample,
        encoder_features=encoder_features,
        decoder_features=decoder_features
    )
    if not (encoder_features or decoder_features):
        seg_pred = seg_pred.argmax(1).unsqueeze(1)
    return seg_pred


def remap(seg_pred, ignore=255):
    if 'nusc' in WEIGHTS.lower():
        mapping = {0: 0, 13: 1, 2: 2, 7: 3, 17: 4, 20: 5, 8: 6, 12: 7, 26: 8, 14: 9, 22: 10, 11: 11, 6: 12, 27: 13,
                   10: 14, 19: 15, 24: 16, 9: 17, 4: 18}
    else:
        mapping = {0: 0, 12: 1, 15: 2, 23: 3, 10: 4, 14: 5, 18: 6, 2: 7, 17: 8, 13: 9, 8: 10, 3: 11, 27: 12, 4: 13,
                   25: 14, 24: 15, 6: 16, 22: 17, 28: 18}
    h, w = seg_pred.shape[-2:]
    seg_pred_remap = np.ones((h, w), dtype=np.uint8) * ignore
    for pseudo, gt in mapping.items():
        whr = seg_pred == pseudo
        seg_pred_remap[whr] = gt
    return seg_pred_remap


def merge_images(images):
    images = [im.convert('RGB') for im in images]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im


def create_model(cuda, resnet=False):
    if resnet:
        raise NotImplementedError('ResNet18+FPN is not yet supported.')

    weights_path = WEIGHTS
    variant_path = '{}_variant{}.yml'.format(weights_path, '_gpu' if cuda else '')

    print('Use weights {}'.format(weights_path))
    print('Load variant from {}'.format(variant_path))
    variant = yaml.load(
        open(variant_path, "r"), Loader=yaml.FullLoader
    )

    # TODO: parse hyperparameters
    window_size = variant['inference_kwargs']["window_size"]
    window_stride = variant['inference_kwargs']["window_stride"]
    im_size = variant['inference_kwargs']["im_size"]

    net_kwargs = variant["net_kwargs"]
    if not resnet:
        net_kwargs['decoder']['dropout'] = 0.

    # TODO: create model
    if resnet:
        model = PanopticFPN(arch=net_kwargs['backbone'], pretrain=net_kwargs['pretrain'], n_cls=net_kwargs['n_cls'])
    else:
        model = create_segmenter(net_kwargs)

    # TODO: load weights
    print('Load weights from {}'.format(weights_path))
    weights = torch.load(weights_path, map_location=torch.device('cpu'))['model']
    model.load_state_dict(weights, strict=True)

    model.eval()
    if cuda:
        model = model.cuda()

    return model, window_size, window_stride, im_size


def get_transformations(im_size):
    trans_list = [transforms.ToTensor()]

    if im_size != 1024:
        trans_list.append(transforms.Resize(im_size))

    trans_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return transforms.Compose(trans_list)


def predict(input_img_path, model, im_size, window_size, window_stride, cuda, alpha):
    input_img_pil = Image.open(input_img_path)
    transform = get_transformations(im_size)
    input_img = transform(input_img_pil)
    input_img = torch.unsqueeze(input_img, 0)
    if cuda:
        input_img = input_img.cuda()

    with torch.no_grad():
        batch_size = 8 if cuda else 1
        segmentation = segment_segmenter(input_img, model, window_size, window_stride,
                                         batch_size=batch_size).squeeze().detach().cpu()
        segmentation_remap = remap(segmentation)

    drawing_pseudo = colorize_one(segmentation_remap)
    drawing_cs = map2cs(segmentation_remap)

    drawing_pseudo = transforms.ToPILImage()(drawing_pseudo).resize(input_img_pil.size)
    drawing_cs = transforms.ToPILImage()(drawing_cs).resize(input_img_pil.size)

    drawing_blend_pseudo = blend_images(input_img_pil, drawing_pseudo, alpha=alpha)
    drawing_blend_cs = blend_images(input_img_pil, drawing_cs, alpha=alpha)

    return input_img_pil, drawing_blend_pseudo, drawing_blend_cs


@click.command(help="")
@click.option("--input-path", type=str, default=None,
              help="Path to an image file or to a directory directory with images.")
@click.option("--output-dir", type=str, default=None,
              help="Path to an output directory for saving the outputs. Optional.")
@click.option("--alpha", type=float, default=0.3, help="Value of alpha used for blending.")
@click.option("--cuda", default=False, is_flag=True, help='Use GPU. It is strongly recommended to do so.')
def main(input_path, output_dir, alpha, cuda):
    # download resources
    download_weights()
    model, window_size, window_stride, im_size = create_model(cuda)

    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.isfile(input_path):
        input_paths = [input_path]
    elif os.path.isdir(input_path):
        input_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if
                       os.path.isfile(os.path.join(input_path, f))]
    else:
        raise Exception("Path to an image or to directory with images must be given in the '--input-path' argument.")

    for path in tqdm(input_paths):
        fname = os.path.split(path)[-1]

        input_img_pil, drawing_blend_pseudo, drawing_blend_cs = predict(path, model, im_size, window_size,
                                                                        window_stride, cuda, alpha)

        if output_dir is not None:
            fname, ext = fname.split('.')

            fname_pseudo = '{}_pseudo.jpg'.format(fname)
            fname_cs = '{}_cs.jpg'.format(fname)
            fname_all = '{}_all.jpg'.format(fname)

            output_path_pseudo = os.path.join(output_dir, fname_pseudo)
            output_path_cs = os.path.join(output_dir, fname_cs)
            output_path_all = os.path.join(output_dir, fname_all)

            drawing_blend_pseudo.save(output_path_pseudo)
            drawing_blend_cs.save(output_path_cs)
            drawing_merged = merge_images([input_img_pil, drawing_blend_pseudo, drawing_blend_cs])
            drawing_merged.save(output_path_all)


if __name__ == '__main__':
    main()
