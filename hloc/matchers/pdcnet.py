from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from einops import rearrange
from easydict import EasyDict as edict

from ..utils.base_model import BaseModel

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '../../third_party'))
from DenseMatching.model_selection import select_model 



class PDCNet(BaseModel):
    default_conf = {
        'weights': 'outdoor',
        'match_threshold': 0.2,
        'max_num_matches': None,
        'model_path': None,  # should be specified
    }
    required_inputs = [
        'image0',
        'image1'
    ]

    def _init(self, conf):
        conf = edict(conf)  # convert to edict
        network, estimate_uncertainty = select_model(
            'PDCNet_plus', pre_trained_model_type, conf, args.optim_iter, local_optim_iter,
            path_to_pre_trained_models=args.path_to_pre_trained_models)

    def _forward(self, data):
        # For consistency with hloc pairs, we refine kpts in image0!
        rename = {
            'keypoints0': 'keypoints1',
            'keypoints1': 'keypoints0',
            'image0': 'image1',
            'image1': 'image0',
            'mask0': 'mask1',
            'mask1': 'mask0',
        }
        data_ = {rename[k]: v for k, v in data.items()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self.net(data_)

        scores = pred['confidence']

        top_k = self.conf['max_num_matches']
        if top_k is not None and len(scores) > top_k:
            keep = torch.argsort(scores, descending=True)[:top_k]
            pred['keypoints0'], pred['keypoints1'] =\
                pred['keypoints0'][keep], pred['keypoints1'][keep]
            scores = scores[keep]

        # Switch back indices
        pred = {(rename[k] if k in rename else k): v for k, v in pred.items()}
        pred['scores'] = scores
        del pred['confidence']
        
        return pred