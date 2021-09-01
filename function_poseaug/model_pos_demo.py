from __future__ import print_function, absolute_import, division

import time
from tokenize import Double

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data_loader import PoseDataSet
from progress.bar import Bar
from utils.data_utils import fetch
from utils.loss import mpjpe, p_mpjpe, compute_PCK, compute_AUC
from utils.utils import AverageMeter


####################################################################
# ### evaluate p1 p2 pck auc dataset with test-flip-augmentation
####################################################################
def evaluate_demo(inputs, model_pos_eval, device, summary=None, thorax_relative=0):
    
    # Switch to evaluate mode
    model_pos_eval.eval()

    outputs_3d_stack = []

    num_poses = inputs.shape[0]
    inputs_2d = torch.Tensor(inputs)

    # Measure data loading time
    inputs_2d = inputs_2d.to(device)

    with torch.no_grad():
        outputs_3d = model_pos_eval(inputs_2d.view(num_poses, -1)).view(num_poses, -1, 3).cpu()
        if thorax_relative == 0:
            outputs_3d_stack = list(outputs_3d)
            #python3 run_demo.py --posenet_name 'transformer' --keypoints gt --evaluate 'checkpoint/ckpt_best_h36m_p1.pth.tar' --thorax_relative 0 --track 1 --video 0

        # caculate the relative position.
        elif thorax_relative == 1:
            outputs_3d = outputs_3d[:, :, :] - outputs_3d[:, 8:9, :]  # the output is relative to the 8 joint
            outputs_3d_stack = list(outputs_3d)
            #python3 run_demo.py --posenet_name 'transformer' --keypoints gt --evaluate 'checkpoint/ckpt_best_h36m_p1.pth.tar' --thorax_relative 1 --track 1 --video 0
    
 
    return outputs_3d_stack
