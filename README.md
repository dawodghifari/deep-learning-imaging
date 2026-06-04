# Deep Learning for Imaging — Denoising, Classification & Detection

Three PyTorch projects from Intelligent Visual Signal Understanding (ELEC5304, University of Sydney, 2026).

## 1. Image Denoising Network

End-to-end denoising pipeline on 400 grayscale 180×180 images: dataset handling, noise modelling, network design, training loop, and quantitative evaluation (PSNR/SSIM).

![Denoising results](assets/denoising_results.png)

## 2. CIFAR-10 Classification Network

10-class image classifier with an 80/20 train/test split: data augmentation (random crop/flip), CNN architecture design, training with device-portable code (CUDA → Apple MPS → CPU fallback), and test-set evaluation.

## 3. Customised Faster R-CNN — PASCAL VOC 2007 Object Detection

Fine-tuned and customised Faster R-CNN for 20-class object detection on PASCAL VOC 2007, using the official train/val/test splits — covering architecture design, training, mAP evaluation, and systematic error analysis.

## Repo contents

| Path | Description |
|---|---|
| `notebooks/image_denoising.ipynb` | Denoising pipeline |
| `notebooks/cifar10_classification.ipynb` | CIFAR-10 classifier |
| `notebooks/voc2007_faster_rcnn.ipynb` | Faster R-CNN detection |
| `docs/` | Project reports (incl. 12-page Faster R-CNN paper) |

## Run it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Datasets download automatically inside the notebooks (CIFAR-10/VOC via torchvision; denoising set via in-notebook command).

## Context

Group projects with Hilal Chamtie, Darin Li, and Akshar Hossain.
