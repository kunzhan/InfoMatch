import ipdb

import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.cifar_cutmix_w import DATASET_GETTERS
from utils import AverageMeter, accuracy, l1_norm
from dataset.mix_1 import mixup_soft, mixup_hard, cutmix_soft, cutmix_hard

from dist_helper import setup_distributed

logger = logging.getLogger(__name__)
A = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
f = open('./log/'+A+'.log','w')
logging.basicConfig(filename='./log/'+A+'.log', level=logging.INFO)
best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))
        # shutil.copyfile(file1,file2)，file1，file2是两个字符串，表示两个文件路径。
        # file1是需要复制的源文件的文件路径，file2是新的文件路径，注意：这个函数只能复制文件，不能复制文件夹

def set_seed(args):
    random.seed(args.seed) # python seed
    os.environ['PYTHONHASHSEED'] = str(args.seed) # 设置python哈希种子，for certain hash-based operations (e.g., the item order in a set or a dict）。seed为0的时候表示不用这个feature，也可以设置为整数。 有时候需要在终端执行，到脚本实行可能就迟了。
    np.random.seed(args.seed) # If you or any of the libraries you are using rely on NumPy, 比如Sampling，或者一些augmentation。 哪些是例外可以看https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(args.seed) # 为当前CPU设置随机种子。 pytorch官网倒是说(both CPU and CUDA)
    torch.cuda.manual_seed(args.seed) # 为当前GPU设置随机种子
    # torch.cuda.manual_seed_all(args.seed) # 使用多块GPU时，均设置随机种子
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True # 设置为True时，cuDNN使用非确定性算法寻找最高效算法
    # torch.backends.cudnn.enabled = True # pytorch使用CUDANN加速，即使用GPU加速

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    # 通过 warmup 获取余弦时刻表
    # 创建一个学习率的计划，该学习率从优化器中设置的初始 lr 线性减小到 0，经过一段预热期后，学习率从 0 逐步变化到优化器中设置的初始 lr 的余弦值。
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):  # interleave，交错
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
    # ([-1, size] + s[1:]) = ([-1, size, s[1:]])
    # 作用是将第一个维度的部分顺序进行调换
    # s[0]->(s[0]/size,size)->(size,s[0]/size)->s[0]


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
    # 作用是将第一个维度的部分顺序进行调换
    # s[0]->(size,s[0]/size)->(s[0]/size,size)->s[0]


@torch.no_grad()
def cal_time_p_and_p_model(logits_x_ulb_w, time_p, p_model, label_hist):
    prob_w = torch.softmax(logits_x_ulb_w, dim=1) 
    max_probs, max_idx = torch.max(prob_w, dim=-1)
    if time_p is None:
        time_p = max_probs.mean()
    else:
        #ipdb.set_trace()
        time_p = time_p * 0.999 +  max_probs.mean() * 0.001
    if p_model is None:
        p_model = torch.mean(prob_w, dim=0)
    else:
        p_model = p_model * 0.999 + torch.mean(prob_w, dim=0) * 0.001
    if label_hist is None:
        label_hist = torch.bincount(max_idx, minlength=p_model.shape[0]).to(p_model.dtype) 
        label_hist = label_hist / label_hist.sum()
    else:
        hist = torch.bincount(max_idx, minlength=p_model.shape[0]).to(p_model.dtype) 
        label_hist = label_hist * 0.999 + (hist / hist.sum()) * 0.001
    return time_p,p_model,label_hist


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='3', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='imagenet', type=str,
                        choices=['cifar10', 'cifar100', 'imagenet'],
                        help='dataset name')
    parser.add_argument('--seed', default=2, type=int, # None
                        help="random seed")

    parser.add_argument('--num-labeled', type=int, default=100000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='resnext', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='net name')
    parser.add_argument('--total-steps', default=2**20, type=int,  # 2**20
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,  # 1024
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=1, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--port', default=9080, type=str, help="don't use progress bar")

    args = parser.parse_args()

    # os.environ["MASTER_PORT"] = str(args.port)
    rank, world_size = setup_distributed(port=args.port)
    args.world_size = world_size

    args.out = 'result/'+args.dataset+'@'+str(args.num_labeled)+'_freematch6'
    
    checkpoint_path = os.path.join(args.out, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_path):
        args.resume = checkpoint_path

    global best_acc

    def create_model(args):
        # set model
        if args.arch == 'wideresnet':  # 选择神经网络模型
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            # import models.resnext as models  # resnext模型有bug
            # model = models.build_resnext(cardinality=args.model_cardinality,
            #                              depth=args.model_depth,
            #                              width=args.model_width,
            #                              num_classes=args.num_classes)
            import models.resnext as net
            builder = getattr(net, 'build_ResNet50')(False)
            model = builder.build(num_classes=args.num_classes)

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        # 输出为日志数据,总参数数目
        return model

    local_rank = int(os.environ["LOCAL_RANK"])
    # set device
    # if args.local_rank == -1:
    #     device = torch.device('cuda', args.gpu_id)
    #     args.world_size = 1
    #     args.n_gpu = torch.cuda.device_count()  # 可用GPU数量
    # else:
    #     torch.cuda.set_device(args.local_rank)  # 设定每个进程使用的GPU
    #     device = torch.device('cuda', args.local_rank)  # 根据local_rank，配置当前进程使用的本地模型及GPU，保证每一个进程使用的GPU是一定的
    #     #torch.distributed.init_process_group(backend='nccl')  # 初始化进程组
    #     torch.distributed.init_process_group(backend='gloo')
    #     args.world_size = torch.distributed.get_world_size()  # 获取全局进程数
    #     args.n_gpu = 1

    # args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)
        # logger.debug,logger.info,logger.warn,logger.error,logger.fatal的作用都是把错误信息写到文本日志里
        # 不同的是它们表示的日志级别不同：日志级别由高到底是：fatal,error,warn,info,debug,低级别的会输出高级别的信息，
        # 高级别的不会输出低级别的信息，如等级设为Error的话，warn,info,debug的信息不会输出

    # logger.warning(
    #     f"Process rank: {local_rank}, "
    #     # f"device: {args.device}, "
    #     f"n_gpu: {args.n_gpu}, "
    #     f"distributed training: {bool(local_rank != -1)}, "
    #     f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))  # 输出args的所有参数
    #ipdb.set_trace()

    if args.seed is not None:
        set_seed(args)

    if rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)  # 在以上代码中，传递给os.makedirs()函数的第一个参数是要创建的目录的路径，第二个参数表示如果目录已经存在是否抛出异常，默认为False。
        # args.writer = SummaryWriter(args.out)  # #第一个参数指明 writer 把summary内容 写在哪个目录下

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    elif args.dataset == 'imagenet':
        args.num_classes = 1000
        args.wdecay = 3e-4
        args.arch == 'resnext'
        args.model_cardinality = 8
        args.model_depth = 29
        args.model_width = 64
        # if args.arch == 'wideresnet':
        #     args.model_depth = 28
        #     args.model_width = 8
        # elif args.arch == 'resnext':
        #     args.model_cardinality = 8
        #     args.model_depth = 29
        #     args.model_width = 64

    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()
    #     # Pytorch在分布式训练过程中，对于数据的读取是采用主进程预读取并缓存，然后其它进程从缓存中读取，
    #     # 不同进程之间的同步通信需要通过torch.distributed.barrier()实现

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, '/idata/ImageNet')  # 数据集加载

    # ipdb.set_trace()

    # if args.local_rank == 0:
        # torch.distributed.barrier()  # 不同进程之间的同步通信

    train_sampler = RandomSampler if local_rank == -1 else DistributedSampler  # 采样器
    # train_sampler 是一个用于对train dataset进行采样的采样器对象。根据 args.local_rank 的值是否为 -1（表示非分布式训练），
    # 选择 RandomSampler（随机采样）或 DistributedSampler（分布式采样）作为采样器。如果 args.local_rank 的值为 -1，
    # 则使用 RandomSampler 对数据进行随机采样；否则，使用 DistributedSampler 进行分布式采样
    # 多点分布式采样的意思是：在不同的地方进行数据采集。采集是采样方式，即隔一定时间对同一点数据重复采集。
    # 采集的数据大多是瞬时值，也可是某段时间内的一个特征值。准确的数据测量是数据采集的基础。

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),  # sampler的主要作用是控制样本的采样顺序，并提供样本的索引
        batch_size=args.batch_size,
        num_workers=args.num_workers,  # num_workers 用于设置数据加载过程中使用的子进程数。其默认值为**0**，即在主进程中进行数据加载，而不使用额外的子进程。
        # 在初始化 dataloader 对象时，会根据num_workers创建子线程用于加载数据(主线程数+子线程数=num_workers)。每个worker或者说线程都有自己负责的dataset范围（下面统称worker）
        drop_last=True)  # 设置 drop_last 参数为 True，表示在最后一个批次数据不足时丢弃该批次。

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),  # 在默认情况下，使用的是SequentialSampler，按照数据集的顺序依次提取样本
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    
    # print(len(labeled_dataset),len(unlabeled_dataset),len(labeled_trainloader),len(unlabeled_trainloader))

    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()  # 不同进程之间的同步通信

    model = create_model(args)
    # ipdb.set_trace()
    # if args.local_rank == 0:
    #     torch.distributed.barrier()  # 不同进程之间的同步通信

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    # 不同参数选用不同的weight_decay进行优化
    # 权重衰减(Wight Decay):在标准SGD的情况下，通过对衰减系数做变换，可以将L2正则和Weight Decay看做一样。
    # 即weight_decay参数就是我们L2正则化的λ值。但是在Adam这种自适应学习率算法中两者并不等价，即L2正则！=Weight Decay。
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 不对偏差进行权重衰减
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)
    # momentum 动量：是上一个step的梯度呈指数加权来影响当前梯度
    # nesterov 是否使用Nesterov动量：是个布尔值，并且可对不同参数组设置不同的值。nesterov动量本意是优化了momentum动量

    args.epochs = math.ceil(args.total_steps / args.eval_step)  # 向上取整
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)
    # 创建一个学习率的计划，预热期与预热期之后学习率的变化不同
    # 该学习率从优化器中设置的初始 lr 线性减小到 0，经过一段预热期后，学习率从 0 逐步变化到优化器中设置的初始 lr 的余弦值

    if args.use_ema:  # 指数移动平均
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)  # 后续迭代计算指数移动平均(.update())

    args.start_epoch = 0

    if args.resume:  # 简要输出最优模型，resume:path to latest checkpoint (default: none)
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
            output_device=local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    #ipdb.set_trace()

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, local_rank)

def min_max(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, local_rank):
    if args.amp:  # amp:use 16-bit (mixed) precision through NVIDIA apex AMP
        # 目的就是使N卡用户能够快速实现自动混合精度(auto mixed precision，amp)技术，在原有模型训练代码中添加少量代码(4行代码)就能够使用amp技术来提高模型的训练效率。apex就是一个支持模型训练在pytorch框架下使用混合精度进行加速训练的拓展库。amp的最核心的东西在于低精度Fp16下进行模型训练。
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:  # 全局进程数
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)  # 各个进程之间设置相同的种子数:0
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)  # iter() 函数用来生成迭代器，即通过iter()函数获取这些可迭代对象的迭代器，实际上就是调⽤了可迭代对象的 __ iter __ ⽅法
    unlabeled_iter = iter(unlabeled_trainloader)

    # 初始化
    if args.resume:  # 简要输出最优模型，resume:path to latest checkpoint (default: none)
        checkpoint = torch.load(args.resume)
        time_p = checkpoint['time_p'].cuda() 
        p_model = checkpoint['p_model'].cuda() 
        label_hist = checkpoint['label_hist'].cuda() 
    else:
        p_model = (torch.ones(args.num_classes) / args.num_classes).cuda()  # 自适应局部阈值
        label_hist = (torch.ones(args.num_classes) / args.num_classes).cuda() # 每类概率
        time_p = p_model.mean() # 自适应全局阈值

    model.train()
    # for epoch in range(args.start_epoch, args.epochs):
    #     batch_time = AverageMeter()
    #     data_time = AverageMeter()
    #     losses = AverageMeter()
    #     losses_x = AverageMeter()
    #     losses_u = AverageMeter()
    #     mask_probs = AverageMeter()

    #     if not args.no_progress:
    #         p_bar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])

    #     # Initialize iterators at the beginning of each epoch
    #     labeled_iter = iter(labeled_trainloader)
    #     unlabeled_iter = iter(unlabeled_trainloader)
    #     print(len(unlabeled_iter))
    #     print(len(unlabeled_trainloader))
    #     print(len(labeled_iter))
    #     print(len(labeled_trainloader))

    #     for batch_idx in range(args.eval_step):
    #         # Handle labeled data
    #         if batch_idx % len(labeled_trainloader) == 0:
    #             if args.world_size > 1:
    #                 labeled_epoch += 1
    #                 labeled_trainloader.sampler.set_epoch(labeled_epoch)
    #             labeled_iter = iter(labeled_trainloader)
    #         inputs_x, targets_x = next(labeled_iter)

    #         # Handle unlabeled data
    #         if batch_idx % len(unlabeled_trainloader) == 0:
    #             if args.world_size > 1:
    #                 unlabeled_epoch += 1
    #                 unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
    #             unlabeled_iter = iter(unlabeled_trainloader)
    #         (inputs_u_w, inputs_u_s1, inputs_u_s2, bbox_w, lam_w), _ = next(unlabeled_iter)

    #     # ... remaining part of your training loop ...
    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()  # AverageMeter()函数用来管理一些需要更新的变量,update:计算和存储平均值和当前值
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:  # yes
            p_bar = tqdm(range(args.eval_step),  # eval:评估
                         disable=local_rank not in [-1, 0])
            # tqdm是 Python 进度条库，可以在 Python长循环中添加一个进度提示信息；disable：是否禁用整个进度条，默认是false
        for batch_idx in range(args.eval_step):
            # 下面代码是数据加载和迭代过程
            try:
                # ipdb.set_trace()
                # inputs_x, targets_x = labeled_iter.next()  # 使用 labeled_iter.next() 从 labeled_trainloader 数据加载器中获取下一个批次的数据
                # error occurs ↓
                inputs_x, targets_x = next(labeled_iter)
            except StopIteration:  # 多进程
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                # inputs_x, targets_x = labeled_iter.next()
                # error occurs ↓
                inputs_x, targets_x = next(labeled_iter)

            try:
                # (inputs_u_w, inputs_u_s1, inputs_u_s2, bbox_w, lam_w), _ = unlabeled_iter.next()  # 无标签样本的增强？
                # error occurs ↓
                (inputs_u_w, inputs_u_s1, inputs_u_s2, bbox_w, lam_w), _ = next(unlabeled_iter)
            except StopIteration:  # 多进程
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                # ipdb.set_trace()
                unlabeled_iter = iter(unlabeled_trainloader)
                # (inputs_u_w, inputs_u_s1, inputs_u_s2, bbox_w, lam_w), _ = unlabeled_iter.next()
                # error occurs ↓
                (inputs_u_w, inputs_u_s1, inputs_u_s2, bbox_w, lam_w), _ = next(unlabeled_iter)
            #ipdb.set_trace()
            inputs_x, inputs_u_w, inputs_u_s1, inputs_u_s2, bbox_w, lam_w = inputs_x.cuda(), inputs_u_w.cuda(), inputs_u_s1.cuda(), \
                inputs_u_s2.cuda(), bbox_w.cuda(), lam_w.cuda()
            indices = torch.randperm(inputs_u_w.size(0)) 
            cutmix_w = cutmix_hard(inputs_u_w, indices, bbox_w)

            data_time.update(time.time() - end)
            batch_size = inputs_x.shape[0]
            inputs = interleave(  # 前两个维度顺序调换
                torch.cat((inputs_x, inputs_u_w, cutmix_w, inputs_u_s1, inputs_u_s2)), 4*args.mu+1).cuda()
            targets_x = targets_x.cuda()
            logits = model(inputs)
            logits = de_interleave(logits, 4*args.mu+1)
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_w_cutmix, logits_u_s1, logits_u_s2 = logits[batch_size:].chunk(4)  # .chunk()方法能够按照某维度，对张量进行均匀切分，并且返回结果是原张量的视图。
            # 第一个参数：目标张量，第二个参数：等分的块数，第三个参数：按照的维度。当原张量不能均分时，chunk不会报错，但会返回其他均分的结果
            del logits  # 删除数据，释放显存，之后可用

            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')  # 监督损失

            time_p, p_model, label_hist = cal_time_p_and_p_model(logits_u_w, time_p, p_model, label_hist)

            pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)  # 锐化:降低标签分布的熵；softmax:转换为概率分布

            # 判断最有信心的类别的概率是否大于阈值，大于阈值选为伪标签
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)

            p_cutoff = time_p  # 自适应全局阈值
            p_model_cutoff = p_model / torch.max(p_model,dim=-1)[0]  # 自适应局部阈值用最大值归一化，保证最大阈值
            threshold = p_cutoff * p_model_cutoff[targets_u]
            '''
            if dataset == 'svhn':
                threshold = torch.clamp(threshold, min=0.9, max=0.95)
                #  clamp()函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量。
            '''
            mask = max_probs.ge(threshold).float()
            
            #cutmix_data, _, target_shuffled, _, max_shuffled, lam = cutmix_hard(logits_u_s, targets_u, max_probs)

            pred_u_s1_norm = (logits_u_s1-torch.mean(logits_u_s1, dim=1, keepdim=True)) / torch.std(logits_u_s1, dim=1, keepdim=True) # Normalization
            pred_u_s2_norm = (logits_u_s2-torch.mean(logits_u_s2, dim=1, keepdim=True)) / torch.std(logits_u_s2, dim=1, keepdim=True)
            # lam1 = lam1*0 + 1
            # lam2 = lam2*0 + 1



            # Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean() + \
            #         (lam * F.cross_entropy(logits_cutmix, targets_u, reduction='none') + \
            #             (1-lam) * F.cross_entropy(logits_cutmix, targets_u[index_shuffled], reduction='none')).mean()  # 一致性正则化损失
            Lu = 0.5*(lam_w * F.cross_entropy(logits_u_w_cutmix, targets_u, reduction='none') * mask + \
                        (1-lam_w) * F.cross_entropy(logits_u_w_cutmix, targets_u[indices], reduction='none') * mask[indices]).mean() + \
                            0.25*(F.cross_entropy(logits_u_s1, targets_u, reduction='none') * mask).mean() + \
                                    0.25*(F.cross_entropy(logits_u_s2, targets_u, reduction='none') * mask).mean() + \
                                            0.001*F.mse_loss(pred_u_s1_norm, pred_u_s2_norm)  
            #ipdb.set_trace()

            loss = Lx + 2.0 * Lu 
            # import ipdb
            # ipdb.set_trace()
            torch.distributed.barrier()
            if args.amp:  # 利用amp进行混合精度计算
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)  # 迭代计算指数移动平均
            model.zero_grad()  # 梯度清零

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())  # 掩码率
            if not args.no_progress:  # yes
                # 使用进度条
                p_bar.set_description(": {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg))
                p_bar.update()
        # 输出到log文件
        logger.info(f"{epoch+1}/{args.epochs}")
        logger.info(f"lr:{scheduler.get_last_lr()[0]},data:{data_time.avg},bt:{batch_time.avg},loss:{losses.avg},loss_x:{losses_x.avg},loss_u:{losses_u.avg},mask:{mask_probs.avg}")

        if not args.no_progress:  # yes
            p_bar.close()  # 关闭进度条

        if args.use_ema:
            test_model = ema_model.ema  # 为啥用ema模型测试，所有迭代过程模型的指数移动平均
        else:
            test_model = model

        if local_rank in [-1, 0]:  # For distributed training: local_rank
            test_loss, test_acc = test(args, test_loader, test_model, epoch, local_rank)

            # 记录epoch，记录多个标量
            # args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            # args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            # args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            # args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            # args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            # args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            # 存储模型及参数
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'time_p': time_p, 
                'p_model': p_model, 
                'label_hist': label_hist,
            }, is_best, args.out)
            # 存储模型，若该模型最优，将其复制到最优模型存储

            test_accs.append(test_acc)
            # 输出精度
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    # if args.local_rank in [-1, 0]:
    #     args.writer.close()  # 结束log文件写入


def test(args, test_loader, model, epoch, local_rank):
    batch_time = AverageMeter()  # AverageMeter()函数用来管理一些需要更新的变量，.update:计算和存储平均值的当前值
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:  # yes
        test_loader = tqdm(test_loader,
                           disable=local_rank not in [-1, 0])
        # tqdm是 Python 进度条库，可以在 Python长循环中添加一个进度提示信息；disable：是否禁用整个进度条，默认是false

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()  # 模型评估，作用是不启用 Batch Normalization 和 Dropout

            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])  # inputs.shape[0]覆盖默认参数n=1，添加多个元素，之后验证一下
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:  # yes
                # 使用进度条
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
                '''
                print("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
                '''
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg
    # 预测和标签是交叉熵损失


if __name__ == '__main__':
    main()
