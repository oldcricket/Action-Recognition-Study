import os
import shutil
import time
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import lr_scheduler
import tensorboard_logger
import torchsummary

from utils.utils import (train, validate, build_dataflow, get_augmentor,
                         save_checkpoint, extract_total_flops_params)
from utils.video_dataset import VideoDataSet, VideoDataSetLMDB
from utils.dataset_config import get_dataset_config
from models import build_model

from opts import arg_parser


def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    cudnn.benchmark = True

    num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file = get_dataset_config(
        args.dataset, args.use_lmdb)

    args.num_classes = num_classes

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5

    model, arch_name = build_model(args)
    mean = model.mean(args.modality)
    std = model.std(args.modality)

    # overwrite mean and std if they are presented in command
    if args.mean is not None:
        if args.modality == 'rgb':
            if len(args.mean) != 3:
                raise ValueError("When training with rgb, dim of mean must be three.")
        elif args.modality == 'flow':
            if len(args.mean) != 1:
                raise ValueError("When training with flow, dim of mean must be three.")
        mean = args.mean

    if args.std is not None:
        if args.modality == 'rgb':
            if len(args.std) != 3:
                raise ValueError("When training with rgb, dim of std must be three.")
        elif args.modality == 'flow':
            if len(args.std) != 1:
                raise ValueError("When training with flow, dim of std must be three.")
        std = args.std

    model = model.cuda()
    model.eval()

    if args.threed_data:
        dummy_data = (3, args.groups, args.input_size, args.input_size)
    else:
        dummy_data = (3 * args.groups, args.input_size, args.input_size)

    model_summary = torchsummary.summary(model, input_size=dummy_data)
    torch.cuda.empty_cache()
    
    if args.show_model:
        print(model)
        print(model_summary)
        return 0

    model = torch.nn.DataParallel(model).cuda()

    if args.pretrained is not None:
        print("=> using pre-trained model '{}'".format(arch_name))
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if args.transfer:
            new_dict = {}
            for k, v in checkpoint['state_dict'].items():
                # TODO: a better approach:
                if k.replace("module.", "").startswith("fc"):
                    continue
                new_dict[k] = v
        else:
            new_dict = checkpoint['state_dict']
        model.load_state_dict(new_dict, strict=False)
    else:
        print("=> creating model '{}'".format(arch_name))

    # define loss function (criterion) and optimizer
    train_criterion = nn.CrossEntropyLoss().cuda()
    val_criterion = nn.CrossEntropyLoss().cuda()

    # Data loading code
    video_data_cls = VideoDataSetLMDB if args.use_lmdb else VideoDataSet
    val_list = os.path.join(args.datadir, val_list_name)
    val_augmentor = get_augmentor(False, args.input_size, mean, std, args.disable_scaleup,
                                  threed_data=args.threed_data, version=args.augmentor_ver,
                                  scale_range=args.scale_range)
    val_dataset = video_data_cls(args.datadir, val_list, args.groups, args.frames_per_group,
                                 num_clips=args.num_clips,
                                 modality=args.modality, image_tmpl=image_tmpl,
                                 dense_sampling=args.dense_sampling,
                                 transform=val_augmentor, is_train=False, test_mode=False,
                                 seperator=filename_seperator, filter_video=filter_video)

    val_loader = build_dataflow(val_dataset, is_train=False, batch_size=args.batch_size,
                                workers=args.workers)

    log_folder = os.path.join(args.logdir, arch_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    if args.evaluate:
        logfile = open(os.path.join(log_folder, 'evaluate_log.log'), 'a')
        flops, params = extract_total_flops_params(model_summary)
        print(model_summary)
        val_top1, val_top5, val_losses, val_speed = validate(val_loader, model, val_criterion)
        print(
            'Val@{}: \tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch\tFlops: {}\tParams: {}'.format(
                args.input_size, val_losses, val_top1, val_top5, val_speed * 1000.0, flops, params),
            flush=True)
        print(
            'Val@{}: \tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch\tFlops: {}\tParams: {}'.format(
                args.input_size, val_losses, val_top1, val_top5, val_speed * 1000.0, flops, params),
            flush=True, file=logfile)
        return

    train_list = os.path.join(args.datadir, train_list_name)

    train_augmentor = get_augmentor(True, args.input_size, mean, std, threed_data=args.threed_data,
                                    version=args.augmentor_ver, scale_range=args.scale_range)
    train_dataset = video_data_cls(args.datadir, train_list, args.groups, args.frames_per_group,
                                   num_clips=args.num_clips,
                                   modality=args.modality, image_tmpl=image_tmpl,
                                   dense_sampling=args.dense_sampling,
                                   transform=train_augmentor, is_train=True, test_mode=False,
                                   seperator=filename_seperator, filter_video=filter_video)

    train_loader = build_dataflow(train_dataset, is_train=True, batch_size=args.batch_size,
                                  workers=args.workers)

    sgd_polices = model.parameters()
    optimizer = torch.optim.SGD(sgd_polices, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    if args.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, args.lr_steps[0], gamma=0.1)
    elif args.lr_scheduler == 'multisteps':
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_steps, gamma=0.1)
    elif args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    elif args.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

    best_top1 = 0.0
    tensorboard_logger.configure(os.path.join(log_folder))
    # optionally resume from a checkpoint
    if args.resume:
        logfile = open(os.path.join(log_folder, 'log.log'), 'a')
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_top1 = checkpoint['best_top1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
            except:
                pass
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError("Checkpoint is not found: {}".format(args.resume))
    else:
        if os.path.exists(os.path.join(log_folder, 'log.log')):
            shutil.copyfile(os.path.join(log_folder, 'log.log'), os.path.join(
                log_folder, 'log.log.{}'.format(int(time.time()))))
        logfile = open(os.path.join(log_folder, 'log.log'), 'w')

    command = " ".join(sys.argv)
    print(command, flush=True)
    print(args, flush=True)
    print(model, flush=True)
    print(model_summary, flush=True)

    print(command, file=logfile, flush=True)
    print(args, file=logfile, flush=True)

    if args.resume == '':
        print(model, file=logfile, flush=True)
        print(model_summary, flush=True, file=logfile)

    for epoch in range(args.start_epoch, args.epochs):
        if args.lr_scheduler == 'plateau':
            scheduler.step(val_losses, epoch)
        else:
            scheduler.step(epoch)
        try:
            # get_lr get all lrs for every layer of current epoch, assume the lr for all layers are identical
            lr = scheduler.optimizer.param_groups[0]['lr']
        except:
            lr = None
        # set current learning rate
        # train for one epoch
        train_top1, train_top5, train_losses, train_speed, speed_data_loader, train_steps = \
            train(train_loader, model, train_criterion, optimizer, epoch + 1,
                  display=args.print_freq,
                  label_smoothing=args.label_smoothing, clip_gradient=args.clip_gradient)
        print(
            'Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch\tData loading: {:.2f} ms/batch'.format(
                epoch + 1, args.epochs, train_losses, train_top1, train_top5, train_speed * 1000.0,
                speed_data_loader * 1000.0), file=logfile, flush=True)
        print(
            'Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch\tData loading: {:.2f} ms/batch'.format(
                epoch + 1, args.epochs, train_losses, train_top1, train_top5, train_speed * 1000.0,
                speed_data_loader * 1000.0), flush=True)

        # evaluate on validation set
        val_top1, val_top5, val_losses, val_speed = validate(val_loader, model, val_criterion)
        print(
            'Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch'.format(
                epoch + 1, args.epochs, val_losses, val_top1, val_top5, val_speed * 1000.0),
            file=logfile, flush=True)
        print(
            'Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch'.format(
                epoch + 1, args.epochs, val_losses, val_top1, val_top5, val_speed * 1000.0),
            flush=True)
        # remember best prec@1 and save checkpoint
        is_best = val_top1 > best_top1
        best_top1 = max(val_top1, best_top1)

        save_dict = {'epoch': epoch + 1,
                     'arch': arch_name,
                     'state_dict': model.state_dict(),
                     'best_top1': best_top1,
                     'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict()
                     }

        save_checkpoint(save_dict, is_best, filepath=log_folder)

        if lr is not None:
            tensorboard_logger.log_value('learning-rate', lr, epoch + 1)
        tensorboard_logger.log_value('val-top1', val_top1, epoch + 1)
        tensorboard_logger.log_value('val-loss', val_losses, epoch + 1)
        tensorboard_logger.log_value('train-top1', train_top1, epoch + 1)
        tensorboard_logger.log_value('train-loss', train_losses, epoch + 1)
        tensorboard_logger.log_value('best-val-top1', best_top1, epoch + 1)

    logfile.close()


if __name__ == '__main__':
    main()
