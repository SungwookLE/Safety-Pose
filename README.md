# Safety-Pose: 3D Human Pose Estimation for Automotive Applications
<table style="border:0px">
   <tr>
       <td><img src="assets/demo1.gif" frame=void rules=none></td>
       <td><img src="assets/demo2.gif" frame=void rules=none></td>
   </tr>
</table>

## 1. Installation
The experiments are conducted on Ubuntu 18.04, with Python version 3.7.8, and PyTorch version 1.7.1.

To setup the environment:
```sh
cd Safety-Pose
conda create -n safety-pose python=3.7.11
conda activate safety-pose
pip install -r requirements.txt
```

- Recomment the docker run environment, docker settings are explained below.

## 2. Prepare dataset
* Please refer to [`DATASETS.md`](./DATASETS.md) for the preparation of the dataset files. 

## 3. Run training code  
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

#### Comment:
* For simplicity, hyper-param for different 2D pose settings are the same. If you want to explore better performance for specific setting, please try changing the hyper-param. 
* The GAN training may collapse, change the hyper-param (e.g., random_seed) and re-train the models will solve the problem.

## 4. Run evaluation code

```sh
python3 run_evaluate.py --posenet_name 'transformer' --keypoints gt --evaluate '/path/to/checkpoint'
```
## 5. Run visualization code
```sh
python3 run_visualization.py --posenet_name 'transformer' --keypoints gt --evaluate '/path/to/checkpoint'
```
## 6. Run Real-time demo code
```sh
python3 run_demo.py --posenet_name 'transformer' --keypoints gt --evaluate '/path/to/checkpoint' --video 0
```
This Real-time demo uses [Lightweight-OpenPose](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) 2D net for detecting 2D keypoints.

For inferencing with `safety testing` recorded images by Realsense Camera, you can add --track option with argument 1. `--track` option supports same target tracking in subsequent images or video 
```sh
python3 run_demo.py --posenet_name 'transformer' --keypoints gt --evaluate '/path/to/checkpoint' --track 1 --images data_extra/test_set/testsets/RGB/*.png
```

For 3D plotting coordinates calculated with thorax relative distance, you can add --thorax_relative option with argument 1. `--thorax_relative` option support calculating relative distance from thorax not hip.
```sh
python3 run_demo.py --posenet_name 'transformer' --keypoints gt --evaluate '/path/to/checkpoint' --thorax_relative 1 --track 1 --video 0
```

---

## 7. Docker Run Setting

A. BASE 이미지: docker pull pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

B. base 이미지에서 아래 패키지를 추가 설치하고 docker push 하여 이미지 생성함

 - 아래 외의 것들도 있을 수 있는데 그건, ModuleImport Error 보면서 추가 설치하였음

	```
	apt update
	apt-get install python3-pip #pip 설치
	apt-get install -y git #git 설치
	git clone https://github.com/SungwookLE/Safety-Pose.git

	pip install tensorboardX
	pip install scipy
	pip install opencv-python
	pip install opencv-contrib-python
	apt-get -y install libgl1-mesa-glx
	pip install pycocotools
	apt install libxcb-xinerama0
	pip install celluloid
	apt-get -y install libgtk2.0-dev
	apt-get install -y libxcb-util1
	```

C. docker 이미지 push 함

 - https://hub.docker.com/repository/docker/joker1251/poseaug
 
D. docker 이미지 pull

	```
	docker pull joker1251/poseaug:0.0
	```
	

E. 옳바르게 docker 컨테이너 torch.cuda 연결되었는지 테스트 코드

	``` python
	import torch

	# GPU check
	torch.cuda.is_available()

	# GPU 정보 확인
	torch.cuda.get_device_name(0)

	# 사용 가능한 GPU 개수
	torch.cuda.device_count()

	# 현재 GPU 번호
	torch.cuda.current_device()

	# device 정보 반환
	torch.cuda.device(0)
	```

F. opencv 비디오 연결이 제대로 되는지 확인
 - 테스트 코드
 
	 ```
	 cd workspace
	 python test.py
	 ```

G. 도커 컨테이너 실행 명령어
 - cv2 카메라를 로컬에서 사용하기 위해 옵션 키워드가 많음
 
 
	```
	docker run -p 8888:8888 --privileged --rm -it -v /dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -v /dev/snd:/dev/snd -e="QT_X11_NO_MITSHM=1" --gpus all -v ~/docker:/data joker1251/poseaug:0.0 /bin/bash
	```

H. poseaug Realtime Demo 실행 명령어
	```
	python3 run_demo.py --posenet_name 'transformer' --keypoints gt --evaluate 'checkpoint/ckpt_best_h36m_p1.pth.tar' --track 1 --video 0 --thorax_relative 1
	```

I. Reference
 - 도커 이미지 생성 Ref: https://greeksharifa.github.io/references/2021/06/21/Docker/

---

## 8. Acknowledgements
This repo is created for cooperation on the Hyundai Motor Group AI Competition project, not for commercial use. The repo is forked from [PoseAug](https://github.com/jfzhang95/PoseAug) and our model uses [SemGCN](https://github.com/garyzhao/SemGCN) as backbone. We thank to the authors for releasing their codes.

## End
