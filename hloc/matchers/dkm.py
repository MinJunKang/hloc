from typing import Any, Dict, Optional, Union
from kornia.core import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from einops import rearrange
from kornia.geometry import resize

from ..utils.base_model import BaseModel

import sys
from pathlib import Path

# after installation
import torchvision.transforms as tvf
from .dkm_model import DKMv3_outdoor, DKMv3_indoor


class DKM(BaseModel):
    default_conf = {
        'weights': 'outdoor',
        'conf_thresh': 0.85,
        'mn_thresh': 0.5,  # px (within n pixel, it's inlier), only valid if symmetric is True
        'max_num_matches': None,
        'symmetric': True  # if True, use symmetric matching, this must be true for mutual neighbor check
    }
    required_inputs = [
        'image0',
        'image1'
    ]

    def _init(self, conf):
        self.symmetric = conf['symmetric']
        self.norm_rgb = tvf.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
        if conf['weights'] == 'outdoor':
            self.net = DKMv3_outdoor(symmetric=self.symmetric)
        else:
            self.net = DKMv3_indoor(symmetric=self.symmetric)
            
        self.h_input, self.w_input = self.net.h_resized, self.net.w_resized
        if self.net.upsample_preds:
            self.h_output, self.w_output = 864, 1152  # fixed size upsampling
        else:
            self.h_output, self.w_output = self.h_input, self.w_input

    def _forward(self, data):
        
        # resize and normalize inputs
        img0 = self.norm_rgb(resize(data['image0'], (self.h_input, self.w_input), interpolation='bilinear'))
        img1 = self.norm_rgb(resize(data['image1'], (self.h_input, self.w_input), interpolation='bilinear'))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warp, certainty = self.net.match(img0, img1)
            
        # by confidence and nn threshold, get the inlier matches
        if self.symmetric:
            ori_coord = torch.stack([warp[:,:self.w_output, :2], warp[:,self.w_output:, 2:]])
            warp_coord = torch.stack([warp[:,:self.w_output, 2:], warp[:,self.w_output:, :2]])
            
            ori_coord[..., 1] = (ori_coord[..., 1] + 1 - 1 / self.h_output) * self.h_output / 2
            ori_coord[..., 0] = (ori_coord[..., 0] + 1 - 1 / self.w_output) * self.w_output / 2
            warp_coord[..., 1] = (warp_coord[..., 1] + 1 - 1 / self.h_output) * self.h_output / 2
            warp_coord[..., 0] = (warp_coord[..., 0] + 1 - 1 / self.w_output) * self.w_output / 2
            
            certainty_mask = certainty > self.conf['conf_thresh']
            ori_coord0 = ori_coord[0, certainty_mask[:, :self.w_output]]
            warp_coord0 = warp_coord[0, certainty_mask[:, :self.w_output]]
            
            warp_coord0to1 = warp_coord[1][warp_coord0[:, 1].long(), warp_coord0[:, 0].long()]
            mask_warp = (warp_coord0to1 - ori_coord0).norm(dim=-1) < self.conf['mn_thresh']  # mutual neighbor check
            
            keypoints0, keypoints1 = ori_coord0[mask_warp], warp_coord0[mask_warp]  # xy coordinates
            
            # scaling back to original image size
            hw0_ori, hw1_ori = data['image0'].shape[2:], data['image1'].shape[2:]
            keypoints0[:, 0] *= hw0_ori[1] / self.w_output
            keypoints0[:, 1] *= hw0_ori[0] / self.h_output
            keypoints1[:, 0] *= hw1_ori[1] / self.w_output
            keypoints1[:, 1] *= hw1_ori[0] / self.h_output
            
            certainty0 = certainty[:, :self.w_output]
            scores = certainty0[certainty0 > self.conf['conf_thresh']][mask_warp]
        else:
            ori_coord, warp_coord = warp[..., :2], warp[..., 2:]
            
            ori_coord[..., 1] = (ori_coord[..., 1] + 1 - 1 / self.h_output) * self.h_output / 2
            ori_coord[..., 0] = (ori_coord[..., 0] + 1 - 1 / self.w_output) * self.w_output / 2
            warp_coord[..., 1] = (warp_coord[..., 1] + 1 - 1 / self.h_output) * self.h_output / 2
            warp_coord[..., 0] = (warp_coord[..., 0] + 1 - 1 / self.w_output) * self.w_output / 2
            
            certainty_mask = certainty > self.conf['conf_thresh']
            keypoints0 = ori_coord[certainty_mask]
            keypoints1 = warp_coord[certainty_mask]
            
            # scaling back to original image size
            hw0_ori, hw1_ori = data['image0'].shape[2:], data['image1'].shape[2:]
            keypoints0[:, 0] *= hw0_ori[1] / self.w_output
            keypoints0[:, 1] *= hw0_ori[0] / self.h_output
            keypoints1[:, 0] *= hw1_ori[1] / self.w_output
            keypoints1[:, 1] *= hw1_ori[0] / self.h_output
            scores = certainty[certainty_mask]

        top_k = self.conf['max_num_matches']
        if top_k is not None and len(scores) > top_k:
            keep = torch.argsort(scores, descending=True)[:top_k]
            keypoints0, keypoints1 = keypoints0[keep], keypoints1[keep]
            scores = scores[keep]

        pred = {
            'keypoints0': keypoints0,
            'keypoints1': keypoints1,
            'batch_indexes': torch.zeros(len(keypoints0), dtype=torch.int64),
            'scores': scores
        }
        return pred
