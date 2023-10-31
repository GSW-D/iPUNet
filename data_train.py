import numpy as np
import warnings
import os
#from code.benchmark_scripts.utils import normalize_pc
import torch
import math
import random
import point_aug
import json
from utils import *
warnings.filterwarnings('ignore')


class DataLoader():
    def __init__(self, path_dir, batch_size,split='train', uniform=False, normal_channel=False):
        self.root = path_dir
        self.uniform = uniform
        self.batch_size=batch_size
        self.pts=[]
        self.normal=[]
        self.orita=[]
        self.model_file=[]
        self.xyz_gts=[]

        num=0
        for root, dirs, files in os.walk(path_dir):
            for f in files:
                data = np.loadtxt(os.path.join(root,f))
                data[:,0:3],m,c=pc_normalize(data[:,0:3])
                xyz_gt=data[:,0:6]
                data= np.array(random.sample(data.tolist(),256))
                pc=data[:,0:3]
                nor=data[:,3:6]
                self.xyz_gts.append(xyz_gt)
                self.pts.append(pc)
                self.normal.append(nor)
                self.model_file.append(f)

        
        self.length=len(self.pts)
        self.shuffle=True
        self.reset()
        print('dataset length: ',len(self.pts))
        
    def __len__(self):
        return len(self.pts)

    def reset(self):
        self.idxs = np.arange(0, self.length)
        if self.shuffle:
            np.random.shuffle(self.idxs)
            self.pts=np.array(self.pts)[self.idxs]
            self.normal=np.array(self.normal)[self.idxs]
            self.model_file=np.array(self.model_file)[self.idxs]
            self.xyz_gts= np.array(self.xyz_gts)[self.idxs]

        self.num_batches = (self.length + self.batch_size - 1) // self.batch_size
        self.batch_idx = 0

    def _get_item(self, index):
        point_set=self.pts[index]
        point_nor=self.normal[index]
        point_file=self.model_file[index]
        xyz_gt=self.xyz_gts[index]
        point_set=np.reshape(point_set,(1,point_set.shape[0],point_set.shape[1]))
        point_set = point_aug.jitter_perturbation_point_cloud(point_set,sigma=0.1,clip=0.05)
        point_set=point_set.squeeze()
        point_set=torch.from_numpy(point_set.astype(np.float32))
        point_nor=torch.from_numpy(point_nor.astype(np.float32))
        xyz_gt=torch.from_numpy(xyz_gt.astype(np.float32))

        return point_set.cuda(), point_nor.cuda(),point_file,xyz_gt.cuda()#,norm.astype(np.float32),center.astype(np.float32)#,RGB


    def __getitem__(self, index):
        return self._get_item(index)
