import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision.transforms.functional import InterpolationMode

import presets
import utils

import mlflow
import numpy as np
import torch.distributed as dist

try:
    from apex import amp
except ImportError:
    amp = None


def start_mlflow_on_master():
    if utils.is_main_process():
        tracking_server_uri = "http://10.10.10.9:4499"
        mlflow.set_tracking_uri(tracking_server_uri)
        mlflow.set_experiment("cifar100")
        mlflow.start_run()


def end_mlflow_on_master():
    if utils.is_main_process():
        mlflow.end_run()


def mlflow_log_args(args):
    if utils.is_main_process():
        for k, v in vars(args).items():
            mlflow.log_param(k, v)


def mlflow_log_meters(k, v, epoch):
    if utils.is_main_process():
        mlflow.log_metric(k, v, epoch)


def mlflow_log_artifact(local_path, artifact_path):
    if utils.is_main_process():
        mlflow.log_artifact(local_path, artifact_path)


def train_one_epoch(model,
                    criterion,
                    optimizer,
                    data_loader,
                    device,
                    epoch,
                    iterations,
                    print_freq,
                    apex=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr',
                            utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s',
                            utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    running_loss = []
    for image, target in metric_logger.log_every(data_loader, print_freq,
                                                 header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(),
                             lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size /
                                             (time.time() - start_time))
        running_loss.append(loss.item())
        mlflow_log_meters("batch_loss", loss.item(), iterations)
        mlflow_log_meters("batch_lr", optimizer.param_groups[0]["lr"],
                          iterations)

        iterations += 1

    #MLflow log
    mlflow_log_meters('epoch_loss', np.mean(running_loss), epoch)
    mlflow_log_meters('epoch_lr', optimizer.param_groups[0]["lr"], epoch)


def evaluate(model, criterion, data_loader, device, epoch=-1, print_freq=100):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq,
                                                     header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if epoch != -1:
        mlflow_log_meters('eval_loss', metric_logger.loss.global_avg, epoch)
        mlflow_log_meters('avg_acc1', metric_logger.acc1.global_avg, epoch)
        mlflow_log_meters('avg_acc5', metric_logger.acc5.global_avg, epoch)

    print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'.format(
        top1=metric_logger.acc1, top5=metric_logger.acc5))
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets",
                              "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(args):
    # Data loading code
    print("Loading data")

    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    if args.data_name == 'ImageNet':
        resize_size, crop_size = 256, 224
        interpolation = InterpolationMode.BILINEAR
        if args.model == 'inception_v3':
            resize_size, crop_size = 342, 299
        elif args.model.startswith('efficientnet_'):
            sizes = {
                'b0': (256, 224),
                'b1': (256, 240),
                'b2': (288, 288),
                'b3': (320, 300),
                'b4': (384, 380),
                'b5': (489, 456),
                'b6': (561, 528),
                'b7': (633, 600),
            }
            e_type = args.model.replace('efficientnet_', '')
            resize_size, crop_size = sizes[e_type]
            interpolation = InterpolationMode.BICUBIC
        data_path = ''

        print("Loading training data")
        st = time.time()
        print("Loading dataset_train from {}".format(data_path))
        dataset = torchvision.datasets.ImageNet(
            root=data_path,
            split='train',
            transform=presets.ClassificationPresetTrain(
                crop_size=crop_size,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob))
        print("Took", time.time() - st)

        print("Loading validation data")
        print("Loading dataset_train from {}".format(data_path))
        dataset_test = torchvision.datasets.ImageNet(
            root=data_path,
            split='val',
            transform=presets.ClassificationPresetEvalCifar10(
                crop_size=crop_size,
                resize_size=resize_size,
                interpolation=interpolation))
    elif args.data_name == 'CIFAR10':
        resize_size, crop_size = 32, 32  #TODO:  다시 고민 필요
        interpolation = InterpolationMode.BILINEAR
        data_path = '/host_server/raid/jihunyoon/data/image_classification/'
        print("Loading training data")
        st = time.time()
        print("Loading dataset_train from {}".format(data_path))
        dataset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=True,
            transform=presets.ClassificationPresetTrainCifar10(
                crop_size=crop_size,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob))
        print("Took", time.time() - st)

        print("Loading validation data")
        print("Loading dataset_test from {}".format(data_path))
        dataset_test = torchvision.datasets.CIFAR10(
            root=data_path,
            train=False,
            transform=presets.ClassificationPresetEvalCifar10(
                crop_size=crop_size,
                resize_size=resize_size,
                interpolation=interpolation))
    elif args.data_name == 'CIFAR100':
        resize_size, crop_size = 32, 32  #TODO:  다시 고민 필요
        interpolation = InterpolationMode.BILINEAR
        data_path = '/host_server/raid/jihunyoon/data/image_classification/'
        print("Loading training data")
        st = time.time()
        print("Loading dataset_train from {}".format(data_path))
        dataset = torchvision.datasets.CIFAR100(
            root=data_path,
            train=True,
            transform=presets.ClassificationPresetTrainCifar10(
                crop_size=crop_size,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob))
        print("Took", time.time() - st)

        print("Loading validation data")
        print("Loading dataset_test from {}".format(data_path))
        dataset_test = torchvision.datasets.CIFAR100(
            root=data_path,
            train=False,
            transform=presets.ClassificationPresetEvalCifar10(
                crop_size=crop_size,
                resize_size=resize_size,
                interpolation=interpolation))

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    if args.apex and amp is None:
        raise RuntimeError(
            "Failed to import apex. Please install apex from https://www.github.com/nvidia/apex "
            "to enable mixed-precision training.")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)
    start_mlflow_on_master(
    )  #TODO: Even if training is interrupted, MLflow run is set as finished.
    mlflow_log_args(args)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True
    dataset, dataset_test, train_sampler, test_sampler = load_data(args)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size,
                                              sampler=train_sampler,
                                              num_workers=args.workers,
                                              pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=args.batch_size,
                                                   sampler=test_sampler,
                                                   num_workers=args.workers,
                                                   pin_memory=True)

    print("Creating model")
    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()

    opt_name = args.opt.lower()
    if opt_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif opt_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        eps=0.0316,
                                        alpha=0.9)
    else:
        raise RuntimeError(
            "Invalid optimizer {}. Only SGD and RMSprop are supported.".format(
                args.opt))

    if args.apex:
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level=args.apex_opt_level)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=args.lr_step_size,
                                                   gamma=args.lr_gamma)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    global iterations
    iterations = 1
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device,
                        epoch, iterations, args.print_freq, args.apex)
        lr_scheduler.step()
        evaluate(model,
                 criterion,
                 data_loader_test,
                 device=device,
                 epoch=epoch)
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args
            }
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint, os.path.join(args.output_dir, 'checkpoint.pth'))

            mlflow_log_artifact(
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)),
                'model')
            mlflow_log_artifact(
                os.path.join(args.output_dir, 'checkpoint.pth'), 'model')
        #if args.mlflow_log_model:  #TODO: Bug
        #mlflow.pytorch.log_model(model_without_ddp, f'model_epoch{epoch}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    end_mlflow_on_master()


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='PyTorch Classification Training', add_help=add_help)

    parser.add_argument('--data-name', help='dataset')
    parser.add_argument('--model', default='resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs',
                        default=90,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j',
                        '--workers',
                        default=16,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--opt', default='sgd', type=str, help='optimizer')
    parser.add_argument('--lr',
                        default=0.1,
                        type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        metavar='M',
                        help='momentum')
    parser.add_argument('--wd',
                        '--weight-decay',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size',
                        default=30,
                        type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma',
                        default=0.1,
                        type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq',
                        default=10,
                        type=int,
                        help='print frequency')
    parser.add_argument('--output-dir', help='path where to save')
    parser.add_argument('--mlflow-log-model',
                        action='store_true',
                        help='log model on MLflow')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help=
        "Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument('--auto-augment',
                        default=None,
                        help='auto augment policy (default: None)')
    parser.add_argument('--random-erase',
                        default=0.0,
                        type=float,
                        help='random erasing probability (default: 0.0)')

    # Mixed precision training parameters
    parser.add_argument('--apex',
                        action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument(
        '--apex-opt-level',
        default='O1',
        type=str,
        help='For apex mixed precision training'
        'O0 for FP32 training, O1 for mixed precision training.'
        'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
    )

    # distributed training parameters
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url',
                        default='env://',
                        help='url used to set up distributed training')

    return parser


if __name__ == "__main__":

    args = get_args_parser().parse_args()
    main(args)
