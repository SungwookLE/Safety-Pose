from __future__ import print_function, absolute_import, division

import os
import os.path as path
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from function_baseline.config import get_parse_args
from function_baseline.data_preparation import data_preparation
from function_baseline.model_pos_preparation import model_pos_preparation

from function_poseaug.model_pos_eval import evaluate, evaluate_safety
from torch.utils.data import DataLoader


def main(args):
    print('==> Using settings {}'.format(args))
    stride = args.downsample
    cudnn.benchmark = True
    device = torch.device("cuda")

    print('==> Loading dataset...')
    data_dict = data_preparation(args)

    print("==> Creating model...")
    model_pos = model_pos_preparation(args, data_dict['dataset'], device)

    # Check if evaluate checkpoint file exist:
    assert path.isfile(args.evaluate), '==> No checkpoint found at {}'.format(args.evaluate)
    print("==> Loading checkpoint '{}'".format(args.evaluate))
    ckpt = torch.load(args.evaluate)
    try:
        model_pos.load_state_dict(ckpt['state_dict'])
    except:
        model_pos.load_state_dict(ckpt['model_pos'])

    print('==> Inferencing...')
    error_3dhp_p1, error_3dhp_p2, pred_3dhp = evaluate(data_dict['3DHP_test'], model_pos, device, flipaug='_flip')
    print('3DHP: Protocol #1   (MPJPE) overall average: {:.2f} (mm)'.format(error_3dhp_p1))
    print('3DHP: Protocol #2 (P-MPJPE) overall average: {:.2f} (mm)'.format(error_3dhp_p2))

    error_safety_p1, error_safety_p2, pred_safety = evaluate_safety(data_dict['safety_test'], model_pos, device, flipaug='_flip')
    print('Safety: Protocol #1   (MPJPE) overall average: {:.2f} (mm)'.format(error_safety_p1))
    print('Safety: Protocol #2 (P-MPJPE) overall average: {:.2f} (mm)'.format(error_safety_p2))

    from common.viz import show3DposePair, show3Dpose, show2Dpose
    import matplotlib.pyplot as plt
    from celluloid import Camera
    import time
    from common.data_loader import PoseDataSet, PoseBuffer

    print('==> 3DHP Visualization...')

    mpi3d_npz = np.load('data_extra/test_set/test_3dhp.npz')
    tmp = mpi3d_npz

    pred_3dhp_dataloader = DataLoader(PoseBuffer(pred_3dhp, [tmp['pose2d']]),
                              batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
    now = time.time()

    fig = plt.figure()
    ax1=fig.add_subplot(1,2,1,projection='3d')
    ax1.set_title('Pair-3D')

    ax2=fig.add_subplot(1,2,2,projection='3d')
    ax2.set_title('Predicted 3D')

    camera = Camera(fig)

    for src, pred in zip(data_dict['3DHP_test'], pred_3dhp_dataloader):
        output_3d, output_2d = src
        pred_3d_, _ = pred

        assert( len(output_3d) == len(pred_3d_))

        for out, pre in zip(output_3d, pred_3d_):
            show3DposePair(out, pre, ax =ax1)
            show3Dpose(pre, ax =ax2)
            camera.snap()
    
            if (time.time() > now+7):
                break

    animation = camera.animate(interval=50, blit=True)

    animation.save(
        '3dhp_show3D.mp4',
        dpi=100,
        savefig_kwargs={
            'frameon': False,
            'pad_inches': 'tight'
        }
    )
    print('==> Safety Visualization...')

    safety_npz = np.load('data_extra/test_set/test_safety.npz')
    tmp = safety_npz
    pred_safety_loader = DataLoader(PoseBuffer(pred_safety, [tmp['pose2d']]),
                               batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers, pin_memory=True)

    
    now = time.time()

    fig2 = plt.figure()
    ax2_1=fig2.add_subplot(1,2,1,projection='3d')
    ax2_1.set_title('Pair-3D')

    ax2_2=fig2.add_subplot(1,2,2,projection='3d')
    ax2_2.set_title('Predicted 3D')

    camera = Camera(fig2)

    for src, pred in zip(data_dict['safety_test'], pred_safety_loader):
        output_3d, output_2d = src
        pred_3d_, _ = pred

        assert( len(output_3d) == len(pred_3d_))

        for out, pre in zip(output_3d, pred_3d_):
            show3DposePair(out, pre, ax =ax2_1)
            show3Dpose(pre, ax =ax2_2)
            camera.snap()
    
            if (time.time() > now+7):
                break

    animation = camera.animate(interval=50, blit=True)

    animation.save(
        'safety_show3D.mp4',
        dpi=100,
        savefig_kwargs={
            'frameon': False,
            'pad_inches': 'tight'
        }
    )

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