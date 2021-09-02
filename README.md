# Safety-Pose: 3D Human Pose Estimation for automotive application
<table style="border:0px">
   <tr>
       <td><img src="assets/demo1.gif" frame=void rules=none></td>
       <td><img src="assets/demo2.gif" frame=void rules=none></td>
   </tr>
</table>

## Installation
The experiments are conducted on Ubuntu 18.04, with Python version 3.7.8, and PyTorch version 1.7.1.

To setup the environment:
```sh
cd Safety-Pose
conda create -n safety-pose python=3.7.11
conda activate safety-pose
pip install -r requirements.txt
```

## Prepare dataset
* Please refer to [`DATASETS.md`](./DATASETS.md) for the preparation of the dataset files. 

## Run training code  
* There are 8 experiments in total (4 for baseline training, 4 for PoseAug training), including four 2D pose settings (Ground Truth, CPN, DET, HR-Net).
* You can also train other pose estimators ([SemGCN](https://github.com/garyzhao/SemGCN), [SimpleBaseline](https://github.com/una-dinosauria/3d-pose-baseline), [ST-GCN](https://github.com/vanoracai/Exploiting-Spatial-temporal-Relationships-for-3D-Pose-Estimation-via-Graph-Convolutional-Networks), [VideoPose](https://github.com/facebookresearch/VideoPose3D)). Please refer to [PoseAug](https://github.com/jfzhang95/PoseAug).
* The training procedure contains two steps: pretrain the baseline models and then train these baseline models with PoseAug.  

To pretrain the baseline model, 
```sh
python3 run_baseline.py --note pretrain --dropout 0 --lr 2e-2 --epochs 100 --posenet_name 'transformer' --checkpoint './checkpoint/pretrain_baseline' --keypoints gt
python3 run_baseline.py --note pretrain --dropout 0 --lr 2e-2 --epochs 100 --posenet_name 'transformer' --checkpoint './checkpoint/pretrain_baseline' --keypoints cpn_ft_h36m_dbb
python3 run_baseline.py --note pretrain --dropout 0 --lr 2e-2 --epochs 100 --posenet_name 'transformer' --checkpoint './checkpoint/pretrain_baseline' --keypoints detectron_ft_h36m
python3 run_baseline.py --note pretrain --dropout 0 --lr 2e-2 --epochs 100 --posenet_name 'transformer' --checkpoint './checkpoint/pretrain_baseline' --keypoints hr
```
To train the baseline model with PoseAug:
```sh
python3 run_poseaug.py --note poseaug --dropout 0 --posenet_name 'transformer' --lr_p 1e-3 --checkpoint './checkpoint/poseaug' --keypoints gt
python3 run_poseaug.py --note poseaug --dropout 0 --posenet_name 'transformer' --lr_p 1e-3 --checkpoint './checkpoint/poseaug' --keypoints cpn_ft_h36m_dbb
python3 run_poseaug.py --note poseaug --dropout 0 --posenet_name 'transformer' --lr_p 1e-3 --checkpoint './checkpoint/poseaug' --keypoints detectron_ft_h36m
python3 run_poseaug.py --note poseaug --dropout 0 --posenet_name 'transformer' --lr_p 1e-3 --checkpoint './checkpoint/poseaug' --keypoints hr
```
All the checkpoints, evaluation results and logs will be saved to `./checkpoint`. You can use tensorboard to monitor the training process:
```sh
cd ./checkpoint/poseaug
tensorboard --logdir=/path/to/eventfile
```

### Comment:
* For simplicity, hyper-param for different 2D pose settings are the same. If you want to explore better performance for specific setting, please try changing the hyper-param. 
* The GAN training may collapse, change the hyper-param (e.g., random_seed) and re-train the models will solve the problem.

## Run evaluation code

```sh
python3 run_evaluate.py --posenet_name 'transformer' --keypoints gt --evaluate '/path/to/checkpoint'
```
## Run visualization code
```sh
python3 run_visualization.py --posenet_name 'transformer' --keypoints gt --evaluate '/path/to/checkpoint'
```
## Run Real-time demo code
```sh
python3 run_demo.py --posenet_name 'transformer' --keypoints gt --evaluate '/path/to/checkpoint' --video 0
```
This Real-time demo uses [Lightweight-OpenPose](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) 2D net for detecting 2D keypoints.

For inferencing with `safety testing` image(realsense) data:
```sh
python3 run_demo.py --posenet_name 'transformer' --keypoints gt --evaluate '/path/to/checkpoint' --track 1 --images data_extra/test_set/testsets/RGB/*.png
```

For 3D plotting coordinates calculated with thorax relative distance, you can add --thorax_relative option with argument 1.
```sh
python3 run_demo.py --posenet_name 'transformer' --keypoints gt --evaluate '/path/to/checkpoint' --thorax_relative 1 --track 1 --video 0
```

## Acknowledgements
This repo is created for cooperation on the Hyundai Motor Group AI Competition project, not for commercial use. The repo is forked from [PoseAug](https://github.com/jfzhang95/PoseAug) and our model uses [SemGCN](https://github.com/garyzhao/SemGCN) as backbone. We thank to the authors for releasing their codes.
