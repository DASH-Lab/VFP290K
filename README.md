# VFP290K: A Large-Scale Benchmark Dataset for Vision-based Fallen Person Detection

This repository is the official documentation & implementation of [VFP290K: A Large-Scale Benchmark Dataset for Vision-based Fallen Person Detection](https://openreview.net/forum?id=y2AbfIXgBK3). 

![VFP290K](./images/teaser.png)

## Requirements

Our pretrained models except YOLO are based on [MMdetection2](https://github.com/open-mmlab/mmdetection) detection framework. You can donwload coco-pretrained models for the transfer learning.

Download our VFP290K dataset in here: [VFP290K](https://sites.google.com/view/dash-vfp300k/download).

## Environments
CUDA=11.1\
python=3.7\
CUDNN=7.6.5

## MMdetection-based models
#### 1. Install Pytorch
We use pytorch=1.8.0 from this [link](https://pytorch.org/get-started/previous-versions/).

#### 2. Install MMDetection & pkgs
```setup
pip install openmim
mim install mmdet
pip install future tensorboard
pip install -r requirements.txt
```

#### 3. Prepare all preprocessed data for training and inference.
```bash
python make_label.py --data_root_dir <VFP directory>
```

#### 4. Download the checkpoints files
You can find checkpoints files in [the official repository](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md)
and **put them into ./checkpoints**

#### 5. Set a config file. 
We prepare all config files used in our experiments in "configs/VFP290K".\
Set your labels.txt and VFP290K data directory.\
(classes= "\<YOUR labels.txt DIRECTORY\>", data_root = "\<YOUR DATA DIRECTORY\>")
    
#### 6. Running Benchmark or desired experiment
To train and evaluate the model(s) in the paper, run this command:
- single gpu training
    ```train
    python tools/train.py <config> --gpu-ids <device> 
    ```
    <config> and <device> indicate path of the config file and gpu id, respectively. This example is for train faster R-CNN model on gpu 0.\
    ex) python tools/train.py configs/VFP290K/faster_rcnn_r50_1x_benchmark.py --gpu-ids 0
    
- multi gpu training
    ```multi gpu training
    bash ./tools/dist_train.sh <config> <num_gpu> 
    ```
    <num_gpu> is a number of gpus to use. This example is for train faster R-CNN model with 4 gpus.\
    ex) bash ./tools/dist_train.sh configs/VFP290K/faster_rcnn_r50_1x_benchmark.py 4 
- test
   After train the model, you can evaluate the result. 
    ```eval
    python tools/test.py <config> <weight> --eval bbox --gpu-ids <device>
    ```
    <weight> is the path of the trained model weight.\
    ex) python tools/test.py configs/VFP290K/faster_rcnn_r50_1x_benchmark.py work_dirs/faster_rcnn_r50_1x_benchmark/latest.pth --eval bbox --gpu-ids 1

## YOLOv5
#### 1. Generate .txt files for yolo.
```bash
cd yolov5
python configs/data_refactoring.py --data_root_dir <{YOUR DATA ROOT DIRECTORY}/yolov5>
```

#### 2. Change the configuration.
E.g) configs/benchmark.yaml\
        train: /media/data1/VFP290K/VFP290K/yolov5/benchmark/train/image\
        val: /media/data1/VFP290K/VFP290K/yolov5/benchmark/val/image
        
#### 2. Training.
Training process is exactly same with official code.
E.g)
```bash
python train.py --img-size 640 --epochs 100 --data configs/benchmark.yaml --batch-size 48 --cfg ./models/yolov5x.yaml --device 0,1 --workers 8 
```
You can train your model by using this script.
    
#### 5. Testing.
Along with the training, test process is also needed to evaluate our model.
```
python test.py --weights runs/train/exp<your_exp_num>/weights/best.pt --data data/test.yaml --batch-size 48 --img-size 640 --conf-thres 0.5 --iou-thres 0.5 --device 0,1
```        
        
        
## Pre-trained Models
You can download all the pretrained models [here](https://drive.google.com/drive/folders/1HT8z-iF3jJRsD4uvfDDKLEExE7N0MqrV)

## Results
Our model achieves the following performance on benchmark:
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow">Method</th>
    <th class="tg-c3ow" colspan="3">Two-Stage</th>
    <th class="tg-c3ow" colspan="3">One-Stage</th>
    <th class="tg-c3ow">Transformer<br>-based</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">Model</td>
    <td class="tg-c3ow">Faster R-CNN</td>
    <td class="tg-c3ow">Cascade R-CNN</td>
    <td class="tg-c3ow">DetectoRS</td>
    <td class="tg-c3ow">RetinaNet</td>
    <td class="tg-c3ow">YOLO3</td>
    <td class="tg-c3ow">YOLO5</td>
    <td class="tg-c3ow">DETR</td>
  </tr>
  <tr>
    <td class="tg-c3ow">mAP</td>
    <td class="tg-c3ow">0.732</td>
    <td class="tg-c3ow">0.751</td>
    <td class="tg-c3ow">0.746</td>
    <td class="tg-c3ow">0.750</td>
    <td class="tg-c3ow">0.590</td>
    <td class="tg-c3ow">0.741</td>
    <td class="tg-c3ow">0.605</td>
  </tr>
  <tr>
    <td class="tg-c3ow">AP_50</td>
    <td class="tg-c3ow">0.873</td>
    <td class="tg-c3ow">0.874</td>
    <td class="tg-c3ow">0.866</td>
    <td class="tg-c3ow">0.910</td>
    <td class="tg-c3ow">0.813</td>
    <td class="tg-c3ow">0.838</td>
    <td class="tg-c3ow">0.868</td>
  </tr>
  <tr>
    <td class="tg-c3ow">AP_75</td>
    <td class="tg-c3ow">0.799</td>
    <td class="tg-c3ow">0.811</td>
    <td class="tg-c3ow">0.797</td>
    <td class="tg-c3ow">0.811</td>
    <td class="tg-c3ow">0.670</td>
    <td class="tg-c3ow">0.784</td>
    <td class="tg-c3ow">0.687</td>
  </tr>
</tbody>
</table>

Our model achieves the following performance on Background:
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
    <td class="tg-c3ow" colspan="3">Street </td>
    <td class="tg-c3ow" colspan="3">Park</td>
    <td class="tg-c3ow" colspan="3">Building </td>
  </tr>
  <tr>
    <td class="tg-c3ow">Faster R-CNN</td>
    <td class="tg-c3ow">mAP<br>AP_50<br>AP_75</td>
    <td class="tg-c3ow">0.742<br>0.910<br>0.829</td>
    <td class="tg-c3ow">0.732<br>0.860<br>0.809</td>
    <td class="tg-c3ow">0.616<br>0.828<br>0.723</td>
    <td class="tg-c3ow">0.620<br>0.786<br>0.690</td>
    <td class="tg-c3ow">0.706<br>0.857<br>0.768</td>
    <td class="tg-c3ow">0.517<br>0.705<br>0.588</td>
    <td class="tg-c3ow">0.748<br>0.876<br>0.813</td>
    <td class="tg-c3ow">0.847<br>0.957<br>0.920</td>
    <td class="tg-c3ow">0.702<br>0.821<br>0.791</td>
  </tr>
  <tr>
    <td class="tg-c3ow">RetinaNet</td>
    <td class="tg-c3ow">mAP<br>AP_50<br>AP_75</td>
    <td class="tg-c3ow">0.770<br>0.922<br>0.843</td>
    <td class="tg-c3ow">0.743<br>0.861<br>0.804</td>
    <td class="tg-c3ow">0.654<br>0.811<br>0.730</td>
    <td class="tg-c3ow">0.664<br>0.830<br>0.720</td>
    <td class="tg-c3ow">0.737<br>0.888<br>0.791</td>
    <td class="tg-c3ow">0.587<br>0.752<br>0.647</td>
    <td class="tg-c3ow">0.828<br>0.932<br>0.901</td>
    <td class="tg-c3ow">0.851<br>0.960<br>0.918</td>
    <td class="tg-c3ow">0.804<br>0.915<br>0.875</td>
  </tr>
  <tr>
    <td class="tg-c3ow">YOLOv3</td>
    <td class="tg-c3ow">mAP<br>AP_50<br>AP_75</td>
    <td class="tg-c3ow">0.610<br>0.817<br>0.689</td>
    <td class="tg-c3ow">0.510<br>0.664<br>0.600</td>
    <td class="tg-c3ow">0.284<br>0.400<br>0.336</td>
    <td class="tg-c3ow">0.416<br>0.578<br>0.468</td>
    <td class="tg-c3ow">0.537<br>0.759<br>0.632</td>
    <td class="tg-c3ow">0.282<br>0.421<br>0.315</td>
    <td class="tg-c3ow">0.610<br>0.817<br>0.689</td>
    <td class="tg-c3ow">0.664<br>0.824<br>0.784</td>
    <td class="tg-c3ow">0.671<br>0.831<br>0.790</td>
  </tr>
  <tr>
    <td class="tg-c3ow">YOLOv5</td>
    <td class="tg-c3ow">mAP<br>AP_50<br>AP_75</td>
    <td class="tg-c3ow">0.669<br>0.783<br>0.729</td>
    <td class="tg-c3ow">0.671<br>0.745<br>0.719</td>
    <td class="tg-c3ow">0.226<br>0.335<br>0.266</td>
    <td class="tg-c3ow">0.398<br>0.465<br>0.428</td>
    <td class="tg-c3ow">0.692<br>0.776<br>0.727</td>
    <td class="tg-c3ow">0.209<br>0.335<br>0.266</td>
    <td class="tg-c3ow">0.675<br>0.743<br>0.727</td>
    <td class="tg-c3ow">0.802<br>0.848<br>0.836</td>
    <td class="tg-c3ow">0.606<br>0.707<br>0.679</td>
  </tr>
</tbody>
</table>

Our model achieves the following performance on light conditions and camera heights:
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow">Backbone</th>
    <th class="tg-c3ow">Training<br></th>
    <th class="tg-c3ow">Day</th>
    <th class="tg-c3ow">Night</th>
    <th class="tg-c3ow">Day</th>
    <th class="tg-c3ow">Night</th>
    <th class="tg-c3ow">Low</th>
    <th class="tg-c3ow">High</th>
    <th class="tg-c3ow">Low</th>
    <th class="tg-c3ow">High</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">Test</td>
    <td class="tg-c3ow" colspan="2">Day</td>
    <td class="tg-c3ow" colspan="2">Night</td>
    <td class="tg-c3ow" colspan="2">Low</td>
    <td class="tg-c3ow" colspan="2">High</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Faster R-CNN</td>
    <td class="tg-c3ow">mAP<br>AP_50<br>AP_75</td>
    <td class="tg-c3ow">0.767<br>0.917<br>0.843</td>
    <td class="tg-c3ow">0.632<br>0.826<br>0.808</td>
    <td class="tg-c3ow">0.523<br>0.714<br>0.572</td>
    <td class="tg-c3ow">0.559<br>0.783<br>0.609</td>
    <td class="tg-c3ow">0.700<br>0.898<br>0.808</td>
    <td class="tg-c3ow">0.573<br>0.760<br>0.669</td>
    <td class="tg-c3ow">0.561<br>0.749<br>0.636</td>
    <td class="tg-c3ow">0.729<br>0.896<br>0.817</td>
  </tr>
  <tr>
    <td class="tg-c3ow">RetinaNet</td>
    <td class="tg-c3ow">mAP<br>AP_50<br>AP_75</td>
    <td class="tg-c3ow">0.779<br>0.932<br>0.848</td>
    <td class="tg-c3ow">0.667<br>0.856<br>0.741</td>
    <td class="tg-c3ow">0.534<br>0.747<br>0.567</td>
    <td class="tg-c3ow">0.566<br>0.785<br>0.620</td>
    <td class="tg-c3ow">0.702<br>0.903<br>0.792</td>
    <td class="tg-c3ow">0.610<br>0.818<br>0.695</td>
    <td class="tg-c3ow">0.596<br>0.780<br>0.669</td>
    <td class="tg-c3ow">0.739<br>0.909<br>0.817</td>
  </tr>
  <tr>
    <td class="tg-c3ow">YOLOv3</td>
    <td class="tg-c3ow">mAP<br>AP_50<br>AP_75</td>
    <td class="tg-c3ow">0.615<br>0.874<br>0.728</td>
    <td class="tg-c3ow">0.432<br>0.630<br>0.490</td>
    <td class="tg-c3ow">0.299<br>0.545<br>0.306</td>
    <td class="tg-c3ow">0.415<br>0.635<br>0.451</td>
    <td class="tg-c3ow">0.567<br>0.808<br>0.678</td>
    <td class="tg-c3ow">0.375<br>0.606<br>0.414</td>
    <td class="tg-c3ow">0.349<br>0.530<br>0.394</td>
    <td class="tg-c3ow">0.563<br>0.800<br>0.653</td>
  </tr>
  <tr>
    <td class="tg-c3ow">YOLOv5</td>
    <td class="tg-c3ow">mAP<br>AP_50<br>AP_75</td>
    <td class="tg-c3ow">0.794<br>0.888<br>0.842</td>
    <td class="tg-c3ow">0.343<br>0.447<br>0.384</td>
    <td class="tg-c3ow">0.392<br>0.517<br>0.416</td>
    <td class="tg-c3ow">0.414<br>0.561<br>0.442</td>
    <td class="tg-c3ow">0.590<br>0.752<br>0.680</td>
    <td class="tg-c3ow">0.412<br>0.542<br>0.465</td>
    <td class="tg-c3ow">0.350<br>0.448<br>0.394</td>
    <td class="tg-c3ow">0.718<br>0.843<br>0.781</td>
  </tr>
</tbody>
</table>

## Contributing
This repository is copyrighted under GPLv3 license 
