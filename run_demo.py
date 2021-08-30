from __future__ import print_function, absolute_import, division
from operator import pos
from common.h36m_dataset import Human36mDataset

import os
import os.path as path
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from function_baseline.config import get_parse_args
from function_baseline.data_preparation import data_preparation
from function_baseline.model_pos_preparation import model_pos_preparation
from function_poseaug.model_pos_demo import evaluate_demo
from net_2d.demo import net_2d_init, run_demo
from data_extra.convert_data_openpose import openpose_to_h36m, show2Dpose, show3Dpose

def main(args):
    print('==> Using settings {}'.format(args))
    stride = args.downsample
    cudnn.benchmark = True
    device = torch.device("cuda")

    print('==> 2D Inferencing...')

    net, frame_provider = net_2d_init()
    pose_keypoints, recorded_imgs = run_demo(net, frame_provider, 256, False, 0, 1)
    pose_keypoints, no_detected_indexs = openpose_to_h36m(pose_keypoints, recorded_imgs[0].shape)

    print("==> 3D Creating model...")
    skeleton_info = Human36mDataset('data/data_3d_h36m.npz')
    model_pos = model_pos_preparation(args,skeleton_info, device)

    # Check if evaluate checkpoint file exist:
    assert path.isfile(args.evaluate), '==> No checkpoint found at {}'.format(args.evaluate)
    print("==> Loading checkpoint '{}'".format(args.evaluate))
    ckpt = torch.load(args.evaluate)
    try:
        model_pos.load_state_dict(ckpt['state_dict'])
    except:
        model_pos.load_state_dict(ckpt['model_pos'])

    pred_demo = evaluate_demo(pose_keypoints, model_pos, device)

    import matplotlib.pyplot as plt
    from celluloid import Camera
    import time

    print('==> Demo Visualization...')
    print(" -> 2D keypoint : {}".format(len(pose_keypoints)))
    print(" -> 3D pred_demo : {}".format(len(pred_demo)))

    now = time.time()
    fig = plt.figure()
    ax1=fig.add_subplot(1,2,1)
    ax1.set_title('Input image + 2D keypoints')
    ax2=fig.add_subplot(1,2,2, projection='3d')
    ax2.set_title('Predicted 3D keypoints')

    camera = Camera(fig)

    for pred in zip(pred_demo, pose_keypoints, recorded_imgs, no_detected_indexs):
        pred_3d, pred_2d, img, no2d_detected = pred

        show2Dpose(img, pred_2d, no2d_index=no2d_detected, ax=ax1, add_labels=False)
        show3Dpose(pred_3d, no2d_index=no2d_detected, ax =ax2)

        camera.snap()
        if (time.time() > now+7):
            break

    animation = camera.animate(interval=50, blit=True)
    animation.save(
        'demo_video.mp4',
        dpi=100,
        savefig_kwargs={
            'frameon': False,
            'pad_inches': 'tight'
        }
    )

    del camera, frame_provider

    return #왜 세그멘테이션 에러 나지 (8/28) del camera
    #python3 run_demo.py --posenet_name 'transformer' --keypoints gt --evaluate 'checkpoint/ckpt_best_h36m_p1.pth.tar'
    
if __name__ == '__main__':
    args = get_parse_args()
    # fix random
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    # copy from #https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True

    main(args)