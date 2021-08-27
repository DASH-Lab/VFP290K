# VFP290K: A Large-Scale Benchmark Dataset for Vision-based Fallen Person Detection

This repository is the official documentation & implementation of [VFP290K: A Large-Scale Benchmark Datasetfor Vision-based Fallen Person Detection](https://openreview.net/forum?id=y2AbfIXgBK3). 

![VFP290K](./images/teaser.jpg)

## Requirements

Our pretrained models except YOLO are based on [MMdetection2](https://github.com/open-mmlab/mmdetection) detection framework.
#### 1. Install mmcv-full
```setup
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```
Please replace `{cu_version}` and `{torch_version}` in the url to your desired one.

#### 2. clone VFP290K repository
```setup
git clone https://git hub.com/DASH-Lab/VFP290K.git
cd VFP290K-main
pip install -r requirements/build.txt
pip install -v -e .
```
- 모델 구동 환경 설명

## Training & Evaluation

### 1. Benchmark
To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```
```eval
python eval.py --input-data <path_to_data> --alpha 10 --beta 20
```

### 2. Experimental setting (Ablation study for various features)
To train the model(s) based on experimenta setting to demonstrate the perfomance shift in the paper Table.4, run this command:

- Light conditions
```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

- Camera heights
```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```
```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

- Background
```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```
```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

## Pre-trained Models
You can download pretrained models here:
- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

- Pre-trained 모델 넣을지 말지 고민중


## Results
Our model achieves the following performance on  Background:
|   Backbone   |    Training    |     Street     |      Park      |    Building    |     Street     |      Park      |    Building    |     Street     |      Park      |    Building    |
|:------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|:--------------:|
|              |      Test      |     Street                                     |      Park      |                |                |    Building    |                |                |
| Faster R-CNN |       mAP      |      0.742     |      0.732     |      0.616     |      0.620     |      0.706     |      0.517     |      0.748     |      0.847     |      0.702     |
|              | AP_50 \| AP_75 | 0.910 \| 0.829 | 0.860 \| 0.809 | 0.828 \| 0.723 | 0.786 \| 0.690 | 0.857 \| 0.768 | 0.705 \| 0.588 | 0.876 \| 0.813 | 0.957 \| 0.920 | 0.821 \| 0.791 |
|   RetinaNet  |       mAP      |      0.770     |      0.743     |      0.654     |      0.664     |      0.737     |      0.587     |      0.828     |      0.851     |      0.804     |
|              | AP_50 \| AP_75 | 0.922 \| 0.843 | 0.861 \| 0.804 | 0.811 \| 0.730 | 0.830 \| 0.720 | 0.888 \| 0.791 | 0.752 \| 0.647 | 0.932 \| 0.901 | 0.960 \| 0.918 | 0.915 \| 0.875 |
|    YOLOv3    |       mAP      |      0.610     |      0.510     |      0.284     |      0.416     |      0.537     |      0.282     |      0.610     |      0.664     |      0.671     |
|              | AP_50 \| AP_75 | 0.817 \| 0.689 | 0.664 \| 0.600 | 0.400 \| 0.336 | 0.578 \| 0.468 | 0.759 \| 0.632 | 0.421 \| 0.315 | 0.817 \| 0.689 | 0.824 \| 0.784 | 0.831 \| 0.790 |
|    YOLOv5    |       mAP      |      0.669     |      0.671     |      0.226     |      0.398     |      0.692     |      0.209     |      0.675     |      0.802     |      0.606     |
|              | AP_50 \| AP_75 | 0.783 \| 0.729 | 0.745 \| 0.719 | 0.335 \| 0.266 | 0.465 \| 0.428 | 0.776 \| 0.727 | 0.335 \| 0.266 | 0.743 \| 0.727 | 0.848 \| 0.836 | 0.707 \| 0.679 |



## Contributing
This repository is copyrighted under GPLv3 license 
