import argparse
import torch
from torch.autograd import Variable
import os
from datetime import datetime
# from net.bgnet import Net
from utils.tdataloader import get_loader
from utils.utils import clip_gradient, AvgMeter, poly_lr
import torch.nn.functional as F
import numpy as np
from torch.amp import GradScaler, autocast
from net.mynet import Net

file = open("log/BGNet.txt", "a")
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)
torch.backends.cudnn.benchmark = True

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

def validate(val_loader, model, scaler):
    model.eval()
    loss_record = AvgMeter()
    with torch.no_grad():
        for i, pack in enumerate(val_loader, start=1):
            images, gts, edges = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            edges = Variable(edges).cuda()

            with autocast('cuda'):
                lateral_map_3, lateral_map_2, lateral_map_1, edge_map = model(images)

                loss3 = structure_loss(lateral_map_3, gts)
                loss2 = structure_loss(lateral_map_2, gts)
                loss1 = structure_loss(lateral_map_1, gts)
                losse = dice_loss(edge_map, edges)
                loss = loss3 + loss2 + loss1 + 3 * losse

            loss_record.update(loss.data, opt.batchsize)
    return loss_record.avg

def train(train_loader, val_loader, model, optimizer, epoch, best_loss, scaler):
    model.train()

    loss_record3, loss_record2, loss_record1, loss_recorde = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        # ---- data prepare ----
        images, gts, edges = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        edges = Variable(edges).cuda()
        # ---- forward ----
        with autocast('cuda'):
            lateral_map_3, lateral_map_2, lateral_map_1, edge_map = model(images)
            # ---- loss function ----
            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            loss1 = structure_loss(lateral_map_1, gts)
            losse = dice_loss(edge_map, edges)
            loss = loss3 + loss2 + loss1 + 3 * losse
        # ---- backward ----
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        clip_gradient(optimizer, opt.clip)
        # ---- recording loss ----
        loss_record3.update(loss3.data, opt.batchsize)
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record1.update(loss1.data, opt.batchsize)
        loss_recorde.update(losse.data, opt.batchsize)
        # ---- train visualization ----
        if i % 60 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge: {:,.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record3.avg, loss_record2.avg, loss_record1.avg, loss_recorde.avg))
            file.write('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                       '[lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge: {:,.4f}]\n'.
                       format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record3.avg, loss_record2.avg, loss_record1.avg, loss_recorde.avg))

    save_path = 'checkpoints/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 5 == 0 or (epoch + 1) == opt.epoch:
        torch.save(model.state_dict(), save_path + 'BGNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'BGNet-%d.pth' % epoch)
        file.write('[Saving Snapshot:]' + save_path + 'BGNet-%d.pth' % epoch + '\n')

    # 验证模型并保存最优模型
    val_loss = validate(val_loader, model, scaler)
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), save_path + 'BGNet-best.pth')
        print('[Saving Best Model:]', save_path + 'BGNet-best.pth')
        file.write('[Saving Best Model:]' + save_path + 'BGNet-best.pth' + '\n')

    return best_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=25, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=416, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=1, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='/kaggle/input/cod10k-train/TrainDataset', help='path to train dataset')
    parser.add_argument('--val_path', type=str,
                        default='/kaggle/input/cod10k-val/ValDataset', help='path to validation dataset')
    parser.add_argument('--train_save', type=str,
                        default='BGNet')
    opt = parser.parse_args()

    # ---- build models ----
    model = Net().cuda()

    # 使用 DataParallel 包装模型
    model = torch.nn.DataParallel(model)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    # 创建 GradScaler 实例
    scaler = GradScaler()

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/Edge/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    val_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    # 打印使用的 GPU 数量
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")

    # 打印实际使用的 GPU 数量
    if isinstance(model, torch.nn.DataParallel):
        print(f"Model is using {len(model.device_ids)} GPUs")

    print("Start Training")

    best_loss = float('inf')
    for epoch in range(opt.epoch):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        best_loss = train(train_loader, val_loader, model, optimizer, epoch, best_loss, scaler)

    file.close()