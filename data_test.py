import numpy as np
import warnings
import os
import torch
import math
import random
import point_aug
import json
from utils import *
warnings.filterwarnings('ignore')


class DataLoader():
    def __init__(self, path_dir, uniform=False, normal_channel=False):
        self.root = path_dir
        self.uniform = uniform
        self.pts=[]
        self.normal=[]
        self.orita=[]
        self.model_file=[]
        self.xyz_gts=[]
        self.norm=[]
        self.center=[]

        for root, dirs, files in os.walk(path_dir):
            for f in files:
                data=np.loadtxt(os.path.join(root,f))
                pts=data[:,0:3]#np.loadtxt(os.path.join(root,f))
                num_patch=int(pts.shape[0]/120.0)
                fps_idx=farthest_point_sample(pts,num_patch)
                fps_xyz=index_points(pts,fps_idx).squeeze().numpy()
                group_knn_idx=knn_point(256,pts,fps_xyz)
                group_knn_xyz=index_points(pts,group_knn_idx).squeeze().numpy()
                for i in range(num_patch):
                    patch_xyz=group_knn_xyz[i,:,:]
                    file_base,_=os.path.splitext(f)
                    patch_xyz,m,c=pc_normalize(patch_xyz)
                    self.pts.append(patch_xyz)
                    self.norm.append(m)
                    self.center.append(c)
                    self.model_file.append(file_base+'.obj.100.'+str(i)+'.txt')

                    self.xyz_gts.append(patch_xyz)
                    self.normal.append(patch_xyz)
                    self.orita.append(patch_xyz)
                #break
        print('dataset length: ',len(self.pts))

    def __len__(self):
        return len(self.pts)

    def _get_item(self, index):
        point_set=self.pts[index]
        point_nor=self.normal[index]
        point_ori=self.orita[index]
        point_file=self.model_file[index]
        xyz_gt=self.xyz_gts[index]
        norm=self.norm[index]
        center=self.center[index]

        return point_set.astype(np.float32), point_nor.astype(np.float32),point_ori.astype(np.float32),point_file,xyz_gt.astype(np.float32),norm.astype(np.float32),center.astype(np.float32)#,RGB



    def __getitem__(self, index):
        return self._get_item(index)

