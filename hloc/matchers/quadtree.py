from typing import Any, Dict, Optional, Union
from kornia.core import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from einops import rearrange

from ..utils.base_model import BaseModel
from kornia.geometry import resize
from kornia.utils.helpers import map_location_to_cpu

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '../../third_party/QuadTreeAttention/FeatureMatching'))
sys.path.append(str(Path(__file__).parent / '../../third_party/QuadTreeAttention/QuadTreeAttention'))
from src.loftr.backbone import build_backbone
from src.loftr.utils.position_encoding import PositionEncodingSine
from src.loftr.loftr_module import LocalFeatureTransformer, FinePreprocess
from src.loftr.utils.coarse_matching import CoarseMatching
from src.loftr.utils.fine_matching import FineMatching


default_cfg = {
    'backbone_type': 'ResNetFPN',
    'resolution': (8, 2),
    'fine_window_size': 5,
    'fine_concat_coarse_feat': True,
    'resnetfpn': {'initial_dim': 128, 'block_dims': [128, 196, 256]},
    'coarse': {
        'd_model': 256,
        'd_ffn': 256,
        'nhead': 8,
        'layer_names': ['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross'],
        'attention': 'linear',
        'temp_bug_fix': True,
        'block_type': 'quadtree',
        'attn_type': 'B',
        'topks': [16, 8, 8],
    },
    'match_coarse': {
        'thr': 0.2,
        'border_rm': 2,
        'match_type': 'dual_softmax',
        'dsmax_temperature': 0.1,
        'skh_iters': 3,
        'skh_init_bin_score': 1.0,
        'skh_prefilter': False,
        'train_coarse_percent': 0.4,
        'train_pad_num_gt_min': 200,
        'sparse_spvs': False,
    },
    'fine': {'d_model': 128, 'd_ffn': 128, 'nhead': 8, 'layer_names': ['self', 'cross'], 'attention': 'linear', 'block_type': 'loftr'},
}

urls: Dict[str, str] = {}
urls["outdoor"] = "https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_feature_match/outdoor.ckpt"
urls["indoor"] = "https://github.com/Tangshitao/QuadTreeAttention/releases/download/QuadTreeAttention_feature_match/indoor.ckpt"


class QuadTreeModel(nn.Module):
    def __init__(self, pretrained: Optional[str] = 'outdoor', config: Dict[str, Any] = default_cfg):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()
        self.pretrained = pretrained
        if pretrained is not None:
            if pretrained not in urls.keys():
                raise ValueError(f"pretrained should be None or one of {urls.keys()}")

            pretrained_dict = torch.hub.load_state_dict_from_url(urls[pretrained], map_location=map_location_to_cpu)
            self.load_state_dict(pretrained_dict['state_dict'])
        self.eval()

    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        _data: Dict[str, Union[Tensor, int, torch.Size]] = {
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 
            'hw1_i': data['image1'].shape[2:]
        }
        
        hw0_i, hw1_i = torch.tensor(_data['hw0_i']), torch.tensor(_data['hw1_i'])
        if _data['hw0_i'] != _data['hw1_i']:  # cannot handle, padding to be the same
            padflag = True
            hw_max = torch.maximum(hw0_i, hw1_i)
            hw_max = (torch.ceil(hw_max / 32) * 32).long()
            data['image0'] = F.pad(data['image0'], [0, hw_max[1]-hw0_i[1], 0, hw_max[0]-hw0_i[0]])
            data['image1'] = F.pad(data['image1'], [0, hw_max[1]-hw1_i[1], 0, hw_max[0]-hw1_i[0]])
        else:
            hw_max = (torch.ceil(hw0_i / 32) * 32).long()
            if torch.Size(hw_max) != _data['hw0_i']:
                padflag = True
                data['image0'] = F.pad(data['image0'], [0, hw_max[1]-hw0_i[1], 0, hw_max[0]-hw0_i[0]])
                data['image1'] = F.pad(data['image1'], [0, hw_max[1]-hw1_i[1], 0, hw_max[0]-hw1_i[0]])
            else:
                padflag = False
            
        feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
        (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(_data['bs']), feats_f.split(_data['bs'])

        _data.update({
            'hw0_c': feat_c0.shape[2:], 
            'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 
            'hw1_f': feat_f1.shape[2:],
        })

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = self.pos_encoding(feat_c0)
        feat_c1 = self.pos_encoding(feat_c1)
        
        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0 = resize(data['mask0'], _data['hw0_c'], interpolation='nearest').flatten(-2)
        if 'mask1' in data:
            mask_c1 = resize(data['mask1'], _data['hw1_c'], interpolation='nearest').flatten(-2)
        
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
      
        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, _data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, _data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, _data)
        
        # if padding was used, remove the keypoints within padding area
        if padflag:
            mask_pad = (_data['mkpts0_f'][:, 0] < hw0_i[1]) & (_data['mkpts0_f'][:, 1] < hw0_i[0]) & \
                          (_data['mkpts1_f'][:, 0] < hw1_i[1]) & (_data['mkpts1_f'][:, 1] < hw1_i[0])
            for k in ['mkpts0_f', 'mkpts1_f', 'mconf', 'b_ids']:
                _data[k] = _data[k][mask_pad]
        
        rename_keys: Dict[str, str] = {
            "mkpts0_f": 'keypoints0',
            "mkpts1_f": 'keypoints1',
            "mconf": 'confidence',
            "b_ids": 'batch_indexes',
        }
        out: Dict[str, Tensor] = {}
        for k, v in rename_keys.items():
            _d = _data[k]
            if isinstance(_d, Tensor):
                out[v] = _d
            else:
                raise TypeError(f'Expected Tensor for item `{k}`. Gotcha {type(_d)}')
        return out

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)


class QuadTree(BaseModel):
    default_conf = {
        'weights': 'outdoor',
        'match_threshold': 0.2,
        'max_num_matches': None,
    }
    required_inputs = [
        'image0',
        'image1'
    ]

    def _init(self, conf):
        cfg = default_cfg
        cfg['match_coarse']['thr'] = conf['match_threshold']
        self.net = QuadTreeModel(pretrained=conf['weights'], config=cfg)

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