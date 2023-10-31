#from utils import load_geodesics
import hydra
import omegaconf
import logging
logger = logging.getLogger(__name__)
import os
import torch
import numpy as np
import importlib
from utils import AverageMeter, ModelWrapper
import utils
from utils import *
import torch.nn.functional as F
from data_test import DataLoader
import point_aug
from chamfer_distance import chamfer_distance
from models.pointnet2_utils import utils as pointnet2_utils
from models.iPUNet import iPUNetwork

def test(cfg):
    test_dataset = DataLoader(cfg.data.test_data_path, 'val')
    model_impl = iPUNetwork(cfg)
    model = ModelWrapper(model_impl).cuda()

    checkpoint = torch.load(cfg.bestpth)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
 
    iter_num=cfg.Iter_num
    for i, data in enumerate(test_dataset):
        data = [np.array([item]) for item in data]
        with torch.no_grad():
            pc,nor,ori,f,xyz_gt,s,c=data
            pc=torch.from_numpy(pc).cuda()
            input_pts=pc
            for iter in range(iter_num):
                outputs = model(input_pts)
                input_pts=outputs[3].detach()
            pc=np.squeeze(data[0])
            norm=np.squeeze(data[5])
            center=np.squeeze(data[6])


            xyz_pre=np.squeeze(outputs[4].cpu().numpy())
            fps_idx=farthest_point_sample_3D(xyz_pre,16)
            xyz_pre=index_points_3D(xyz_pre,fps_idx)
            xyz_pre=xyz_pre.reshape(-1,3)  
            xyz_pre=xyz_pre*norm+center
    

@hydra.main(config_path='config/config.yaml', strict=False)#@hydra.main(config_path='config', config_name='config')
def main(cfg):
    omegaconf.OmegaConf.set_struct(cfg, False)
    test(cfg)


if __name__ == '__main__':
    main()