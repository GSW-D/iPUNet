import os
import hydra
import torch
import logging
logger = logging.getLogger(__name__)
import omegaconf
import importlib
import numpy as np
from tqdm import tqdm
from utils import AverageMeter, ModelWrapper
import utils
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from data_train import DataLoader
from tensorboardX import SummaryWriter
import time
from models.iPUNet import iPUNetwork


def train(cfg):

    log_dir = os.path.curdir 
    train_data_path= cfg.data.train_data_path
    val_data_path= cfg.data.val_data_path
    test_data_path= cfg.data.test_data_path

    train_dataset = DataLoader(train_data_path,cfg.batch_size, 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

    val_dataset = DataLoader(val_data_path,cfg.batch_size, 'val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    
    model_impl = iPUNetwork(cfg)
    model = ModelWrapper(model_impl).cuda()
    
    
    logger.info('Start training on point upsampling...')
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.00001
    )

    writer = SummaryWriter(cfg.summarywriter)
    criterion = utils.IPU_Criterion()  
    meter = AverageMeter()
    best_loss = 1e10
    start_epoch=0
    iter_num=cfg.Iter_num
    for epoch in range(start_epoch+1,cfg.max_epoch + 1):
        train_iter = tqdm(train_dataloader)
        l0=[]
        l1=[]
        l2=[]
        l3=[]
        l4=[]
        l5=[]
        l6=[]
        l_total=[]
        
        meter.reset()
        model.train()
        for i, data in enumerate(train_iter):
            pc,nor,f,xyz_gt=data
            loss=torch.zeros((1),device=pc.device,requires_grad=True)
            input_pts=pc
            for iter in range(iter_num):
                outputs = model(input_pts)
                loss_all,loss_smooth,loss_nor,loss_nor_step2,loss_ori_nor_step1,reoloss,loss_charm= criterion(data, outputs,input_pts,epoch)
                input_pts=outputs[3].detach()
                loss=loss+loss_all
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_iter.set_postfix(loss=loss.item())
            meter.update(loss.item()/iter_num)

            l0.append(loss_smooth.item())
            l1.append(loss_nor.item())
            l2.append(loss_nor_step2.item())
            l3.append(loss_ori_nor_step1.item())
            l4.append(reoloss.item())
            l5.append(loss_charm.item())
            l6.append(loss_charm.item())

        logger.info(
                f'E: {epoch},  l: {meter.avg},{torch.mean(torch.Tensor(l0)):.5f},{torch.mean(torch.Tensor(l1)):.5f},{torch.mean(torch.Tensor(l2)):.5f},{torch.mean(torch.Tensor(l3)):.5f},{torch.mean(torch.Tensor(l4)):.5f},\
                    {torch.mean(torch.Tensor(l5)):.5f},\
                {torch.mean(torch.Tensor(l6)):.5f}'
            )
        writer.add_scalar('train_loss',meter.avg,epoch)

        model.eval()
        meter.reset()
        val_iter = tqdm(val_dataloader)
        list_cd=[]
        for i, data in enumerate(val_iter):
            with torch.no_grad():
                pc,nor,f,xyz_gt=data
                loss=torch.zeros((1),device=pc.device,requires_grad=True)
                input_pts=pc
                for iter in range(iter_num):
                    outputs = model(input_pts)
                    input_pts=outputs[3]
                    loss_all,loss_smooth,loss_nor,loss_nor_step2,loss_ori_nor_step1,reoloss,loss_charm= criterion(data, outputs,input_pts,epoch)
                    loss=loss+loss_all
                    list_cd.append(loss_charm.item())
                    

            val_iter.set_postfix(loss=loss.item()/iter_num)
            meter.update(loss.item()/iter_num)


        if torch.mean(torch.Tensor(list_cd))< best_loss:
            logger.info("best epoch: {}".format(epoch))
            best_loss = torch.mean(torch.Tensor(list_cd))
            state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss}
            torch.save(state, '/data1/gswei/code/benchmark_scripts/test_path/test_dgcnn_iter10_256_big_k6.pth')

        logger.info(
                f'Epoch: {epoch}, Average Val loss: {meter.avg}, CD: {torch.mean(torch.Tensor(list_cd))}'
            ) 
        writer.add_scalar('test_loss',meter.avg,epoch)


@hydra.main(config_path='config/config.yaml', strict=False)
def main(cfg):
    omegaconf.OmegaConf.set_struct(cfg, False)
    train(cfg)


if __name__ == '__main__':
    main()