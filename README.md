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
Our model achieves the following performance on :
| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

- 실험 결과표 나오면 oliner generator로 생성해서 붙이기
- PR 커브도 넣기


## Contributing
This repository is copyrighted under GPLv3 license 
