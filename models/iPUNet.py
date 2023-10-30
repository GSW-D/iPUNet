import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
#from sklearn.neighbors.kde import KernelDensity
from .pointnet2_utils import unet as unet_utils
import random
import os
from chamfer_distance import chamfer_distance
import matplotlib.pyplot as plt
from .dgcnn import DGCNN
import time
from knn_cuda import KNN


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


class KNN(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k

    def forward(self, xyz1, xyz2):
        dist = square_distance(xyz1, xyz2).sqrt()
        return torch.topk(dist, self.k, dim=-1, largest=False, sorted=False)

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def get_ori_rot90(ori,nor):
    x=ori[:,:,1].mul(nor[:,:,2])-ori[:,:,2].mul(nor[:,:,1])
    y=ori[:,:,2].mul(nor[:,:,0])-ori[:,:,0].mul(nor[:,:,2])
    z=ori[:,:,0].mul(nor[:,:,1])-ori[:,:,1].mul(nor[:,:,0])
    x=x.reshape(x.shape[0],x.shape[1],1)
    y=y.reshape(y.shape[0],y.shape[1],1)
    z=z.reshape(z.shape[0],z.shape[1],1)
    ori_rot90=torch.cat([x,y,z],dim=2).reshape(ori.shape)
    return ori_rot90

def batch_norm(inputs):
    inp=torch.sqrt(torch.sum(inputs.mul(inputs),dim=2)+0.00000001).view(inputs.shape[0],inputs.shape[1],1)
    #print(inp.shape)
    outp=inputs/(inp+0.0000000001)
    return outp

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx


def index_points_feat(points, idx,feat):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long,device=device).view(view_shape).repeat(repeat_shape)
    points[batch_indices, idx, :]=feat
    return points

def rot_pts_3D(pts,oris,nors,pts_feat,grid_unit,sample_num,radiu_grid,knns):
    
    grid_unit=7
    sample_num=48
    radiu_grid=0.3
    group_idx=query_ball_point(radiu_grid,sample_num,pts,pts)
    group_pts=index_points(pts,group_idx)

    group_feat=index_points(pts_feat,group_idx)
    group_feat=group_feat.reshape((-1,sample_num,group_feat.shape[3]))
    

    ori_rot90=batch_norm(get_ori_rot90(oris,nors))
    ori_0=batch_norm(get_ori_rot90(ori_rot90,nors))


    rot=torch.cat([ori_0,ori_rot90,nors],dim=-1).reshape(-1,3,3).permute(0,2,1)
    pts_repeat=pts.repeat(1,1,sample_num).reshape(group_pts.shape)
    local_coorpts=(group_pts-pts_repeat).reshape(-1,sample_num,3)

    local_pts=torch.matmul(local_coorpts[:,:,None],rot[:,None]).reshape(local_coorpts.shape)
    x=torch.round((local_pts[:,:,0]+radiu_grid)/(radiu_grid*2)*(grid_unit-1)).long()
    y=torch.round((local_pts[:,:,1]+radiu_grid)/(radiu_grid*2)*(grid_unit-1)).long()
    z=torch.round((local_pts[:,:,2]+radiu_grid)/(radiu_grid*2)*(grid_unit-1)).long()


    idx=x*grid_unit+y*grid_unit+z
    group_feat_grid=torch.zeros((8,grid_unit,1),device=pts.device,requires_grad=True)
    group_feat_grid=group_feat_grid.repeat(group_feat.shape[0]//8,grid_unit*grid_unit,128).reshape(group_feat.shape[0],grid_unit*grid_unit*grid_unit,128)
    feat3d=index_points_feat(group_feat_grid,idx,group_feat)

    return feat3d,rot,pts_repeat


def rot_offset(offset,rot,pts,grid_unit=7):

    pts_grid=[]
    local_x=pts*0.0
    local_x[:,:,0]=1
    local_y=pts*0.0
    local_y[:,:,1]=1
    for i in range(-30,36,10):
        for j in range(-30,36,10):
            gx=local_x*i*0.01
            gy=local_y*j*0.01
            pts_local=gx+gy#+gz
            pts_grid.append(pts_local)

    pts_grid=torch.cat(pts_grid,dim=2).reshape(-1,grid_unit*grid_unit,3)
    pts_repeat=pts.repeat(1,1,grid_unit*grid_unit).reshape(pts_grid.shape)
    upsample_pts=pts_grid+offset
    rot_2=rot.permute(0,2,1)
    upsample=torch.matmul(upsample_pts[:,:,None],rot_2[:,None]).reshape(pts_grid.shape)+pts_repeat

    return upsample

class iPUNetwork(nn.Module):
    def __init__(self, cfg):
        super(iPUNetwork, self).__init__()
        self.pointconv1=DGCNN(cfg)
        self.unet3d=unet_utils.UNet3D()
        self.mlp0 = nn.Conv1d(128, 3, 1)
        self.mlp1 = nn.Conv1d(128, 3, 1)
        self.KNN=KNN(k=48)

    def forward(self, xyz):
        pointcloud=xyz[:,0:3,:]

        xyz=pointcloud
        pointcloud = pointcloud.permute(0, 2, 1)

        grid_unit1=7
        sample_num=48
        radiu_grid=0.3

        l0_points=self.pointconv1(xyz)

        oris = batch_norm(self.mlp0(l0_points.transpose(1, 2)).transpose(1, 2))#oris.transpose(1, 2).contiguous()#
        nors = batch_norm(self.mlp1(l0_points.transpose(1, 2)).transpose(1, 2))#nors.transpose(1, 2).contiguous()#
        pts=pointcloud[:,:,0:3]
        pts_feat=l0_points#.transpose(1, 2)
        
        feat_3d,rot,pts_repeat=rot_pts_3D(pts,oris,nors,pts_feat,grid_unit1,sample_num,radiu_grid,self.KNN)
        feat_3d=feat_3d.reshape(-1,grid_unit1,grid_unit1,grid_unit1,128).permute(0,4,1,2,3)
        feat_3d_pc,feat_pc=self.unet3d(feat_3d)
        feat_3d=feat_3d_pc.permute(0,2,3,1).reshape(-1,grid_unit1*grid_unit1,3)

        pts_up=rot_offset(feat_3d,rot,pts,grid_unit1)
        index=torch.randint(0,grid_unit1*grid_unit1,(16,),device=pts_feat.device)
        pts_offset=pts_up[:,24,:].reshape(pts.shape[0],-1,3)
        pts_ups=torch.index_select(pts_up,1,index).reshape(pts.shape[0],-1,3)

        return (oris,nors,pts_ups,pts_offset,pts_up)


SaliencyModel = iPUNetwork


