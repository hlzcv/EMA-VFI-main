import os
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse

from Trainer_event import EventModel
from dataset_events import BSERGBEventDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler


device = torch.device('cuda')


def get_learning_rate(step, total_steps):
    if step < 2000:
        mul = step / 2000
        return 2e-4 * mul
    mul = np.cos((step - 2000) / max(total_steps - 2000, 1) * math.pi) * 0.5 + 0.5
    return (2e-4 - 2e-5) * mul + 2e-5


def train(model, local_rank, batch_size, data_root):
    writer = SummaryWriter('log/train_EMAVFI_event') if local_rank == 0 else None

    train_dataset = BSERGBEventDataset('train', data_root)
    train_sampler = DistributedSampler(train_dataset)
    train_data = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )

    val_dataset = BSERGBEventDataset('validation', data_root)
    val_data = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=8)

    steps_per_epoch = len(train_data)
    total_steps = 300 * max(steps_per_epoch, 1)

    step = 0
    nr_eval = 0
    time_stamp = time.time()

    for epoch in range(300):
        train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            imgs, event_feat = batch
            imgs = imgs.to(device, non_blocking=True) / 255.0
            event_feat = event_feat.to(device, non_blocking=True)
            imgs, gt = imgs[:, 0:6], imgs[:, 6:]

            lr = get_learning_rate(step, total_steps)
            _, loss = model.update(imgs, gt, event_feat=event_feat, learning_rate=lr, training=True)

            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            if writer is not None and step % 200 == 1:
                writer.add_scalar('learning_rate', lr, step)
                writer.add_scalar('loss', loss, step)

            if local_rank == 0:
                print(
                    'epoch:{} {}/{} time:{:.2f}+{:.2f} loss:{:.4e}'.format(
                        epoch, i, steps_per_epoch, data_time_interval, train_time_interval, loss
                    )
                )
            step += 1

        nr_eval += 1
        if nr_eval % 3 == 0:
            evaluate(model, val_data, nr_eval, local_rank)
        model.save_model(local_rank)
        dist.barrier()


def evaluate(model, val_data, nr_eval, local_rank):
    writer_val = SummaryWriter('log/validate_EMAVFI_event') if local_rank == 0 else None

    psnr = []
    for _, batch in enumerate(val_data):
        imgs, event_feat = batch
        imgs = imgs.to(device, non_blocking=True) / 255.0
        event_feat = event_feat.to(device, non_blocking=True)
        imgs, gt = imgs[:, 0:6], imgs[:, 6:]

        with torch.no_grad():
            pred, _ = model.update(imgs, gt, event_feat=event_feat, training=False)

        for j in range(gt.shape[0]):
            psnr.append(-10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item()))

    psnr = np.array(psnr).mean() if len(psnr) > 0 else 0.0
    if local_rank == 0:
        print(str(nr_eval), psnr)
        if writer_val is not None:
            writer_val.add_scalar('psnr', psnr, nr_eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--data_root', type=str, required=True, help='root path of BS-ERGB dataset, e.g. G:/bs_ergb')
    parser.add_argument('--pretrained_name', type=str, default='ours', help='ckpt name to initialize RGB pretrained weight')
    args = parser.parse_args()

    torch.distributed.init_process_group(backend='nccl', world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    if args.local_rank == 0 and not os.path.exists('log'):
        os.mkdir('log')

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    model = EventModel(args.local_rank)
    model.load_model(name=args.pretrained_name, rank=args.local_rank)
    train(model, args.local_rank, args.batch_size, args.data_root)
