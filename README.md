<h1 align="center">Welcome to the code for Drive&Segment 👋</h1>

<p align="center">

[comment]: <> (  <a href="https://www.npmjs.com/package/readme-md-generator">)

[comment]: <> (    <img alt="downloads" src="https://img.shields.io/npm/dm/readme-md-generator.svg?color=blue" target="_blank" />)

[comment]: <> (  </a>)
  <a href="TBD">
    <img alt="Project Page" src="https://img.shields.io/badge/Project Page-Open-green.svg" target="_blank" />
  </a>
  <a href="https://github.com/kefranabg/readme-md-generator/blob/master/LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-yellow.svg" target="_blank" />
  </a>
  <a href="https://twitter.com/AVobecky">
    <img alt="Twitter: AVobecky" src="https://badgen.net/badge/icon/twitter?icon=twitter&label" target="_blank" />
  </a>
  <a href="https://huggingface.co/spaces/vobecant/DaS">
    <img alt="Gradio App" src="https://img.shields.io/badge/Gradio App-Open%20In%20Spaces-blue.svg" target="_blank" />
  </a>
</p>

# 🚙📷 Drive&Segment: Unsupervised Semantic Segmentation of Urban Scenes via Cross-modal Distillation

This project hosts the code for inference of the Drive&Segment for unsupervised image segmentation of urban scenes.

> [**Drive&Segment: Unsupervised Semantic Segmentation of Urban Scenes via Cross-modal Distillation**](TBD)            
> Antonin Vobecky, David Hurych, Oriane Siméoni, Spyros Gidaris, Andrei Bursuc, Patrick Pérez, and Josef Sivic
>
> *arXiv preprint ([arXiv XYZ.xyz](TBD))*

![teaser](teaser.png)

## 💫 Highlights

- **A:**
- 📷💥 **Multi-modal training:** During the train time our method takes 📷 images and 💥 LiDAR scans as an input, and
  ... *Mention datasets.*
- 📷 **Image-only inference:** Drive&Segment predicts ...
- 🏆 **State-of-the-art performance:** Our best single model based on Segmenter architecture achieves **X%** in mIoU on
  Cityscapes (without any fine-tuning).
- 🚀 **Gradio Application**: We provide an interactive [Gradio application](https://huggingface.co/spaces/vobecant/DaS)
  so that everyone can try our model.

## 📺 Examples

### **Pseudo** segmentation.

Example of **pseudo** segmentation.

![](sources/video128_blend03_v2_10fps_640px_lanczos.gif)

### Cityscapes segmentation.

Two examples of pseudo segmentation mapped to the 19 ground-truth classes of the Cityscapes dataset by using Hungarian
algorithm.

![](sources/video_stuttgart00_remap_blended03_20fps_crop.gif)
![](sources/video_stuttgart01_remap_blended03_20fps_crop2.gif)

## ✨ Running the models

### 📝 Requirements

Please, refer to `requirements.txt`

### 🚀 Inference

We provide our Segmenter model trained on the nuScenes dataset.

Run

```
python3 inference.py --input-path [path to image/folder with images] --output-dir [where to save the outputs] --cuda
```

where:

- `--input-path` specifies either a path to a single image or a path to a folder with images,
- `output-dir` (optional) specifies the output directory, and
- `--cuda` is flag denoting whether to run the code on the GPU **(strongly recommended)**.

Example: `python3 inference.py --input-path sources/img1.jpeg --output-dir ./outputs --cuda`

### nuScenes model

We provide [weights](https://data.ciirc.cvut.cz/public/projects/2022DriveAndSegment/segmenter_nusc.pth) and config
files ([CPU](https://data.ciirc.cvut.cz/public/projects/2022DriveAndSegment/segmenter_nusc.pth_variant.yml)
/ [GPU](https://data.ciirc.cvut.cz/public/projects/2022DriveAndSegment/segmenter_nusc.pth_variant_gpu.yml)) for the
Segmenter model trained on the nuScenes dataset.

The weights and config files are downloaded automatically when running `inference.py`. Should you prefer to download
them by hand, please place them to the `./weights` folder.

### Waymo Open model

Due to the Waymo Open dataset licence terms, we cannot openly share the trained weights. If you are interested in using
the model trained on the Waymo Open dataset, please register at
the [Waymo Open](https://waymo.com/intl/en_us/dataset-download-terms/) and send the confirmation of your agreement to
the licence terms to the [authors](mailto:antonin.vobecky@cvut.cz).