import torch
import torch.nn.functional as F
device_ids=[3]
import os
import random
import pickle
from tqdm import tqdm
import numpy as np
from chamfer_distance import chamfer_distance
#import chamfer3D.dist_chamfer_3D
from knn_cuda import KNN
import time
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
knn_number=6

BASEDIR = os.path.dirname(os.path.abspath(__file__))
#device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def add_noise(x, sigma=0.015, clip=0.05):
    noise = np.clip(sigma*np.random.randn(*x.shape), -1*clip, clip)
    return x + noise

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    points=np.reshape(points,(1,points.shape[0],points.shape[1]))
    points=torch.from_numpy(points.astype(np.float32))
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # import ipdb; ipdb.set_trace()
    xyz=np.reshape(xyz,(1,xyz.shape[0],xyz.shape[1]))
    xyz=torch.from_numpy(xyz.astype(np.float32))
    B, N, C = xyz.shape
    centroids = torch.zeros([B, npoint], dtype=torch.long)
    distance = torch.ones(B, N)* 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long)
    batch_indices = torch.arange(B, dtype=torch.long)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def square_distance_torch(src, dst):
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

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    xyz=np.reshape(xyz,(1,xyz.shape[0],xyz.shape[1]))
    new_xyz=np.reshape(new_xyz,(1,new_xyz.shape[0],new_xyz.shape[1]))
    xyz=torch.from_numpy(xyz.astype(np.float32))
    new_xyz=torch.from_numpy(new_xyz.astype(np.float32))

    sqrdists = square_distance_torch(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx.numpy()


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc,m,centroid

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # import ipdb; ipdb.set_trace()
    xyz=np.reshape(xyz,(1,xyz.shape[0],xyz.shape[1]))
    xyz=torch.from_numpy(xyz.astype(np.float32))
    B, N, C = xyz.shape
    centroids = torch.zeros([B, npoint], dtype=torch.long)
    distance = torch.ones(B, N)* 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long)
    batch_indices = torch.arange(B, dtype=torch.long)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points_3D(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    #points=np.reshape(points,(1,points.shape[0],points.shape[1]))
    points=torch.from_numpy(points.astype(np.float32))

    #idx=np.reshape(idx,(1,idx.shape[0],idx.shape[1]))
    #idx=torch.from_numpy(idx)

    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points.numpy()

def farthest_point_sample_3D(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # import ipdb; ipdb.set_trace()
    #xyz=np.reshape(xyz,(1,xyz.shape[0],xyz.shape[1]))
    xyz=torch.from_numpy(xyz.astype(np.float32))
    B, N, C = xyz.shape
    centroids = torch.zeros([B, npoint], dtype=torch.long)
    distance = torch.ones(B, N)* 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long)
    batch_indices = torch.arange(B, dtype=torch.long)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def square_distance_cuda(src, dst):
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
    #print(src.shape)

    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_idx(nsample, xyz,xyz_up):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    xyz=np.reshape(xyz,(xyz.shape[0],1,xyz.shape[1]))
    xyz=torch.from_numpy(xyz.astype(np.float32))
    #xyz_up=np.reshape(xyz_up,(1,xyz_up.shape[0],xyz_up.shape[1]))
    xyz_up=torch.from_numpy(xyz_up.astype(np.float32))
    #print(xyz.shape)

    sqrdists = square_distance_cuda(xyz, xyz_up)
    #print(sqrdists.shape)
    group_dis, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    # dist_knn=torch.mean(group_dis,dim=2)
    #print(dist_knn.shape)
    return group_idx.numpy()

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

class ModelWrapper(torch.nn.Module):
    def __init__(self, model_impl) -> None:
        super().__init__()
        self.model_impl = model_impl
    
    def forward(self, pc):
        if isinstance(pc, np.ndarray):
            pc = torch.from_numpy(pc).float()
            #xyz_gt = torch.from_numpy(xyz_gt).float()
        res = self.model_impl(pc.transpose(1,2).cuda())
        return res
    
def batch_norm(inputs):
    inp=torch.sqrt(torch.sum(inputs.mul(inputs),dim=2)+0.00000001).view(inputs.shape[0],inputs.shape[1],1)
    #print(inp.shape)
    outp=inputs/(inp+0.0000000001)
    return outp

def get_ori_rot90(ori,nor):
    device = ori.device
    x=ori[:,1].mul(nor[:,2])-ori[:,2].mul(nor[:,1])
    y=ori[:,2].mul(nor[:,0])-ori[:,0].mul(nor[:,2])
    z=ori[:,0].mul(nor[:,1])-ori[:,1].mul(nor[:,0])
    ori_rot90=torch.ones(ori.shape,device=device)
    ori_rot90[:,0]=x
    ori_rot90[:,1]=y
    ori_rot90[:,2]=z
    #print(ori[0],nor[0],ori_rot90[0])
    # print(ori_rot90[0])
    # orirotnew=torch.cat([x.reshape(1,-1),y.reshape(1,-1),z.reshape(1,-1)],dim=1).reshape(ori.shape)
    # print(orirotnew[0])
    return ori_rot90


def avg_ori(group_ori,group_nor,batch_ori,batch_nor,num_nebor=knn_number):
    batch_size=batch_ori.shape[0]
    npoints=batch_ori.shape[1]
    batch_ori_repeat=batch_ori.repeat(1,1,num_nebor).reshape(group_ori.shape)
    batch_nor_repeat=batch_nor.repeat(1,1,num_nebor).reshape(group_nor.shape)

    nor_diff=torch.sum(group_nor.mul(batch_nor_repeat),dim=3)
    nor_diff=torch.exp(-nor_diff/0.3)*10+1
    zero = torch.zeros_like(nor_diff)+1
    ones = torch.zeros_like(nor_diff)+5
    nor_diff=torch.where(nor_diff<4,zero,ones)

    batch_ori_rot=get_ori_rot90(batch_ori.reshape(-1,3),batch_nor.reshape(-1,3)).reshape(batch_ori.shape)
    batch_ori_rot_repeat=batch_ori_rot.repeat(1,1,num_nebor).reshape(group_nor.shape)
    cos0=F.cosine_similarity(group_ori, batch_ori_repeat,dim=3).reshape(batch_size,npoints,num_nebor,1)
    cos1=F.cosine_similarity(group_ori, batch_ori_rot_repeat,dim=3).reshape(batch_size,npoints,num_nebor,1)
    cos=torch.cat([cos0,cos1],dim=-1).reshape(batch_size,npoints,num_nebor,-1)
    cos=1-torch.abs(cos)
    cos=torch.min(cos,dim=-1)[0]
    

    point_wise_loss=nor_diff.mul(cos)
    loss_ori=torch.mean(point_wise_loss)
    return loss_ori

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
    batch_indices = torch.arange(B, dtype=torch.long,device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def square_distance_index(src, dst):
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
    #print(src.shape)

    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    index_min=torch.min(dist,dim=2)[1]
    return index_min


def get_ori_project(nor_gt,ori_pre):
    d_n=torch.sum(nor_gt.mul(ori_pre),dim=2).view(nor_gt.shape[0],-1,1)
    ori_dn=nor_gt.mul(d_n)
    ori_pro=ori_pre-ori_dn
    ori_pro=batch_norm(ori_pro)
    #print(torch.mean(torch.sum(nor_gt.mul(ori_pro),dim=2)))
    return ori_pro,d_n

class IPU_Criterion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.knn = KNN(knn_number)
        
    def forward(self, inputs, outputs,pts,epoch):
        ori_pre=batch_norm(outputs[0])
        nor_pre=batch_norm(outputs[1])
        xyz_up=outputs[2]
        pts=pts[:,:,0:3].cuda()
        xyz_gt=inputs[3][:,:,0:3].cuda()
        xyz_nor=inputs[3][:,:,3:6].cuda()


        '''ori-loss'''
        index=square_distance_index(pts,xyz_gt)
        nor_gt=index_points(xyz_nor,index)
        ori_pro,d_n=get_ori_project(nor_gt,ori_pre)

        '''smooth the field of knn points'''
        xyz=pts
        _,idx=self.knn(xyz.transpose(1,2).contiguous(),xyz.transpose(1,2).contiguous())
        idx=idx.transpose(1,2)
        group_ori=index_points(ori_pro,idx)
        group_nor=index_points(nor_gt,idx)
        loss_smooth=avg_ori(group_ori,group_nor,ori_pro,nor_gt)
        
        '''normal estimation'''
        loss_nor=torch.mean(1-torch.abs(F.cosine_similarity(nor_gt, nor_pre,dim=2)))
        

        '''stage12_nor loss'''
        index_1=square_distance_index(xyz_up,xyz_gt)
        nor_gt_1=index_points(xyz_nor,index_1)
       

        '''charmer distance loss'''
        xyz_offset=outputs[3]
        dist1_offset,dist2_offset=chamfer_distance(xyz_offset,xyz_gt)#self.loss_fn_charm(xyz_up,xyz_gt)
        loss_dist1_offset=torch.mean(dist1_offset)
        loss_dist2_offset=torch.mean(dist2_offset)
        loss_charm_offset=loss_dist1_offset+loss_dist2_offset

        dist1,dist2=chamfer_distance(xyz_up,xyz_gt)#self.loss_fn_charm(xyz_up,xyz_gt)
        loss_dist1=torch.mean(dist1)
        loss_dist2=torch.mean(dist2)
        loss_charm=loss_dist1+loss_dist2

        
        '''nor_ori cross'''
        loss_nor_ori=torch.mean(torch.abs(torch.sum(ori_pre.mul(nor_gt),dim=2)))

        '''upsample points charmer distace '''
        loss_smooth=loss_smooth
        loss_nor=loss_nor
        loss_nor_ori=loss_nor_ori

        loss_cd=loss_charm+0.4*loss_charm_offset#+weight_refine*refine_loss_charm#+weight_refine*refine_loss_charm#5*loss_charm+loss_charm_121#60*loss_smooth+loss_nor+200*loss_charm+1000*loss_charm_up
        loss=(loss_smooth+loss_nor+0.1*loss_nor_ori)+(200*loss_cd)#+repulsion_loss1#5*(loss_ori)+loss_nor+loss_nor_ori#+loss_smooth#+10*loss_nor_ori#+5*loss_smooth#+50*loss_nor_ori#+loss_smooth#+loss_ori_nor+loss_smooth
        return loss,loss_smooth,loss_nor,loss_charm_offset,loss_nor_ori,loss_charm,loss_charm 