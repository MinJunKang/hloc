import sys, os
import torch
from pathlib import Path

from collections import OrderedDict
from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / '../../third_party'))
from SGMNet.sgmnet import matcher as SGM_Model
from SGMNet.utils import evaluation_utils
from easydict import EasyDict as edict


class SGMNet(BaseModel):
    default_conf = {
        'seed_top_k': [256,256],
        'seed_radius_coe': 0.01,
        'net_channels': 256,
        'layer_num': 9,
        'head': 4,
        'seedlayer': [0,6],
        'use_mc_seeding': True,
        'use_score_encoding': False,
        'conf_bar': [1.11,0.1],
        'sink_iter': [10,100],
        'detach_iter': 1000000,
        'p_th': 0.2,
        'model_dir': None  # should be specified
    }
    required_inputs = [
        'keypoints0', 'descriptors0', 'image_size0', 'scores0',
        'keypoints1', 'descriptors1', 'image_size1', 'scores1',
    ]

    def _init(self, conf):
        conf = edict(conf)  # convert to edict
        
        if conf.model_dir is None:
            raise ValueError('The weight path should be specified.')
        assert(Path(conf.model_dir).is_dir())  # check if the weight path is valid
        
        self.p_th = conf.p_th
        self.net = SGM_Model(conf)
        checkpoint = torch.load(os.path.join(conf.model_dir, 'model_best.pth'))
        #for ddp model
        if list(checkpoint['state_dict'].items())[0][0].split('.')[0]=='module':
            new_stat_dict=OrderedDict()
            for key,value in checkpoint['state_dict'].items():
                new_stat_dict[key[7:]]=value
            checkpoint['state_dict']=new_stat_dict
        self.net.load_state_dict(checkpoint['state_dict'])
        
    def arange_like(self, x, dim: int):
        return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1
        
    def match_p(self, scores):
        max0, max1 = scores.max(2), scores.max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = self.arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = self.arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.p_th)
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        
        return {
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }

    def _forward(self, data):
        
        rename = {
            'keypoints0': 'x1',
            'keypoints1': 'x2',
            'scores0': 'scores1',
            'scores1': 'scores2',
            'image_size0': 'size1',
            'image_size1': 'size2',
            'descriptors0': 'desc1',
            'descriptors1': 'desc2',
        }
        data_ = {rename[k]: v for k, v in data.items() if k in self.required_inputs}
        
        kpts0, kpts1 = data_['x1'], data_['x2']
        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }
        
        device = kpts0.device
        
        norm_x1 = evaluation_utils.normalize_size(kpts0, data_['size1'].to(device))
        norm_x2 = evaluation_utils.normalize_size(kpts1, data_['size2'].to(device))
        
        # pad score at last dimension
        x1 = torch.cat([norm_x1, data_['scores1'][..., None]], dim=-1)
        x2 = torch.cat([norm_x2, data_['scores2'][..., None]], dim=-1)
        desc1 = data_['desc1'].transpose(-1, -2)
        desc2 = data_['desc2'].transpose(-1, -2)
        feed_data = {'x1': x1, 'x2': x2, 'desc1': desc1, 'desc2': desc2}
        
        with torch.no_grad():
            out = self.net(feed_data, test_mode=True)
            
        return self.match_p(out['p'][:, :-1, :-1])