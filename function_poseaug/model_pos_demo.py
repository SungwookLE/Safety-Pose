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
def evaluate_demo(inputs, model_pos_eval, device, summary=None):
    
    # Switch to evaluate mode
    model_pos_eval.eval()

    outputs_3d_stack = []

    num_poses = inputs.shape[0]
    inputs_2d = torch.Tensor(inputs)

    # Measure data loading time
    inputs_2d = inputs_2d.to(device)

    with torch.no_grad():
        outputs_3d = model_pos_eval(inputs_2d.view(num_poses, -1)).view(num_poses, -1, 3).cpu()
        outputs_3d_stack = list(outputs_3d)

        # caculate the relative position.
        outputs_3d = outputs_3d[:, :, :] - outputs_3d[:, :1, :]  # the output is relative to the 0 joint
 
    return outputs_3d_stack
