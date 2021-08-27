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
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow">Backbone</th>
    <th class="tg-c3ow">Training<br></th>
    <th class="tg-c3ow">Street</th>
    <th class="tg-c3ow">Park</th>
    <th class="tg-c3ow">Building</th>
    <th class="tg-c3ow">Street</th>
    <th class="tg-c3ow">Park</th>
    <th class="tg-c3ow">Building</th>
    <th class="tg-c3ow">Street</th>
    <th class="tg-c3ow">Park</th>
    <th class="tg-c3ow">Building</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">Test</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">Street</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">Park</td>
    <td class="tg-c3ow">Park</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">Building</td>
    <td class="tg-c3ow"></td>
  </tr>
  <tr>
    <td class="tg-c3ow">Faster R-CNN</td>
    <td class="tg-c3ow">mAP</td>
    <td class="tg-c3ow">0.742</td>
    <td class="tg-c3ow">0.732</td>
    <td class="tg-c3ow">0.616</td>
    <td class="tg-c3ow">0.620</td>
    <td class="tg-c3ow">0.706</td>
    <td class="tg-c3ow">0.517</td>
    <td class="tg-c3ow">0.748</td>
    <td class="tg-c3ow">0.847</td>
    <td class="tg-c3ow">0.702</td>
  </tr>
  <tr>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">AP_50 | AP_75</td>
    <td class="tg-c3ow">0.910 | 0.829</td>
    <td class="tg-c3ow">0.860 | 0.809</td>
    <td class="tg-c3ow">0.828 | 0.723</td>
    <td class="tg-c3ow">0.786 | 0.690</td>
    <td class="tg-c3ow">0.857 | 0.768</td>
    <td class="tg-c3ow">0.705 | 0.588</td>
    <td class="tg-c3ow">0.876 | 0.813</td>
    <td class="tg-c3ow">0.957 | 0.920</td>
    <td class="tg-c3ow">0.821 | 0.791</td>
  </tr>
  <tr>
    <td class="tg-c3ow">RetinaNet</td>
    <td class="tg-c3ow">mAP</td>
    <td class="tg-c3ow">0.770</td>
    <td class="tg-c3ow">0.743</td>
    <td class="tg-c3ow">0.654</td>
    <td class="tg-c3ow">0.664</td>
    <td class="tg-c3ow">0.737</td>
    <td class="tg-c3ow">0.587</td>
    <td class="tg-c3ow">0.828</td>
    <td class="tg-c3ow">0.851</td>
    <td class="tg-c3ow">0.804</td>
  </tr>
  <tr>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">AP_50 | AP_75</td>
    <td class="tg-c3ow">0.922 | 0.843</td>
    <td class="tg-c3ow">0.861 | 0.804</td>
    <td class="tg-c3ow">0.811 | 0.730</td>
    <td class="tg-c3ow">0.830 | 0.720</td>
    <td class="tg-c3ow">0.888 | 0.791</td>
    <td class="tg-c3ow">0.752 | 0.647</td>
    <td class="tg-c3ow">0.932 | 0.901</td>
    <td class="tg-c3ow">0.960 | 0.918</td>
    <td class="tg-c3ow">0.915 | 0.875</td>
  </tr>
  <tr>
    <td class="tg-c3ow">YOLOv3</td>
    <td class="tg-c3ow">mAP</td>
    <td class="tg-c3ow">0.610</td>
    <td class="tg-c3ow">0.510</td>
    <td class="tg-c3ow">0.284</td>
    <td class="tg-c3ow">0.416</td>
    <td class="tg-c3ow">0.537</td>
    <td class="tg-c3ow">0.282</td>
    <td class="tg-c3ow">0.610</td>
    <td class="tg-c3ow">0.664</td>
    <td class="tg-c3ow">0.671</td>
  </tr>
  <tr>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">AP_50 | AP_75</td>
    <td class="tg-c3ow">0.817 | 0.689</td>
    <td class="tg-c3ow">0.664 | 0.600</td>
    <td class="tg-c3ow">0.400 | 0.336</td>
    <td class="tg-c3ow">0.578 | 0.468</td>
    <td class="tg-c3ow">0.759 | 0.632</td>
    <td class="tg-c3ow">0.421 | 0.315</td>
    <td class="tg-c3ow">0.817 | 0.689</td>
    <td class="tg-c3ow">0.824 | 0.784</td>
    <td class="tg-c3ow">0.831 | 0.790</td>
  </tr>
  <tr>
    <td class="tg-c3ow">YOLOv5</td>
    <td class="tg-c3ow">mAP</td>
    <td class="tg-c3ow">0.669</td>
    <td class="tg-c3ow">0.671</td>
    <td class="tg-c3ow">0.226</td>
    <td class="tg-c3ow">0.398</td>
    <td class="tg-c3ow">0.692</td>
    <td class="tg-c3ow">0.209</td>
    <td class="tg-c3ow">0.675</td>
    <td class="tg-c3ow">0.802</td>
    <td class="tg-c3ow">0.606</td>
  </tr>
  <tr>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">AP_50 | AP_75</td>
    <td class="tg-c3ow">0.783 | 0.729</td>
    <td class="tg-c3ow">0.745 | 0.719</td>
    <td class="tg-c3ow">0.335 | 0.266</td>
    <td class="tg-c3ow">0.465 | 0.428</td>
    <td class="tg-c3ow">0.776 | 0.727</td>
    <td class="tg-c3ow">0.335 | 0.266</td>
    <td class="tg-c3ow">0.743 | 0.727</td>
    <td class="tg-c3ow">0.848 | 0.836</td>
    <td class="tg-c3ow">0.707 | 0.679</td>
  </tr>
</tbody>
</table>


## Contributing
This repository is copyrighted under GPLv3 license 
