import argparse
from models.model_builder import MODEL_TABLE
from utils.dataset_config import DATASET_CONFIG

def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch Action recognition Training')

    # model definition
    parser.add_argument('--backbone_net', default='s3d', type=str, help='backbone network',
                        choices=list(MODEL_TABLE.keys()))
    parser.add_argument('-d', '--depth', default=18, type=int, metavar='N',
                        help='depth of resnet (default: 18)', choices=[18, 34, 50, 101, 152])
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='dropout ratio before the final layer')
    parser.add_argument('--groups', default=16, type=int, help='number of frames')
    parser.add_argument('--frames_per_group', default=1, type=int,
                        help='[uniform sampling] number of frames per group; '
                             '[dense sampling]: sampling frequency')
    parser.add_argument('--alpha', default=2, type=int, metavar='N',
                        help='[blresnet] ratio of channels')
    parser.add_argument('--beta', default=4, type=int, metavar='N',
                        help='[blresnet] ratio of layers')
    parser.add_argument('--fpn_dim', type=int, default=-1, help='enable 3D fpn on 2d net')
    parser.add_argument('--without_t_stride', dest='without_t_stride', action='store_true',
                        help='skip the temporal stride in the model')
    parser.add_argument('--pooling_method', default='max',
                        choices=['avg', 'max'], help='method for temporal pooling method or '
                                                     'which pool3d module')
    parser.add_argument('--dw_t_conv', dest='dw_t_conv', action='store_true',
                        help='[S3D model] enable depth-wise conv for temporal modeling')
    # model definition: temporal model for 2D models
    parser.add_argument('--temporal_module_name', default=None, type=str,
                        help='which temporal aggregation module to use, '
                             'TSM, TAM and NonLocal are only available on 2D model',
                        choices=[None, 'TSM', 'TAM', 'MAX', 'AVG', 'Channel-Max', 'NonLocal',
                                 'TSM+NonLocal', 'TAM+NonLocal'])
    parser.add_argument('--blending_frames', default=3, type=int)
    parser.add_argument('--blending_method', default='sum',
                        choices=['sum', 'max', 'maxnorm'], help='method for blending channels')
    parser.add_argument('--no_dw_conv', dest='dw_conv', action='store_false',
                        help='[2D model] disable depth-wise conv for temporal modeling')
    parser.add_argument('--consensus', default='avg', type=str, help='which consesus to use',
                        choices=['avg', 'trn', 'multitrn'])

    # training setting
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--disable_cudnn_benchmark', dest='cudnn_benchmark', action='store_false',
                        help='Disable cudnn to search the best mode (avoid OOM)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_scheduler', default='cosine', type=str,
                        help='learning rate scheduler',
                        choices=['step', 'multisteps', 'cosine', 'plateau'])
    parser.add_argument('--lr_steps', default=[15, 30, 45], type=float, nargs="+",
                        metavar='LRSteps', help='[step]: use a single value: the periodto decay '
                                                'learning rate by 10. '
                                                '[multisteps] epochs to decay learning rate by 10')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--nesterov', action='store_true',
                        help='enable nesterov momentum optimizer')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--clip_gradient', '--cg', default=None, type=float,
                        help='clip the total norm of gradient before update parameter')
    parser.add_argument('--label_smoothing', default=0.0, type=float, metavar='SMOOTHING',
                        help='label_smoothing against the cross entropy')
    parser.add_argument('--no_imagenet_pretrained', dest='imagenet_pretrained',
                        action='store_false',
                        help='disable to load imagenet pretrained model')
    parser.add_argument('--transfer', action='store_true',
                        help='perform transfer learning, remove weights in the fc '
                             'layer or the original model.')

    # data-related
    parser.add_argument('-j', '--workers', default=18, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--datadir', metavar='DIR', help='path to dataset file list')
    parser.add_argument('--dataset', default='st2stv2',
                        choices=list(DATASET_CONFIG.keys()), help='path to dataset file list')
    parser.add_argument('--threed_data', action='store_true',
                        help='load data in the layout for 3D conv')
    parser.add_argument('--input_size', default=224, type=int, metavar='N', help='input image size')
    parser.add_argument('--disable_scaleup', action='store_true',
                        help='do not scale up and then crop a small region, directly crop the input_size')
    parser.add_argument('--random_sampling', action='store_true',
                        help='perform determinstic sampling for data loader')
    parser.add_argument('--dense_sampling', action='store_true',
                        help='perform dense sampling for data loader')
    parser.add_argument('--augmentor_ver', default='v1', type=str, choices=['v1', 'v2'],
                        help='[v1] TSN data argmentation, [v2] resize the shorter side to `scale_range`')
    parser.add_argument('--scale_range', default=[256, 320], type=int, nargs="+",
                        metavar='scale_range', help='scale range for augmentor v2')
    parser.add_argument('--modality', default='rgb', type=str, help='rgb or flow',
                        choices=['rgb', 'flow'])
    parser.add_argument('--mean', type=float, nargs="+",
                        metavar='MEAN', help='mean, dimension should be 3 for RGB, 1 for flow')
    parser.add_argument('--std', type=float, nargs="+",
                        metavar='STD', help='std, dimension should be 3 for RGB, 1 for flow')
    parser.add_argument('--skip_normalization', action='store_true',
                        help='skip mean and std normalization, default use imagenet`s mean and std.')
    parser.add_argument('--use_lmdb', action='store_true',
                        help='use lmdb instead of jpeg.')
    # logging
    parser.add_argument('--logdir', default='', type=str, help='log path')
    parser.add_argument('--print-freq', default=100, type=int,
                        help='frequency to print the log during the training')
    parser.add_argument('--show_model', action='store_true', help='show model summary')

    # for testing and validation
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--num_crops', default=1, type=int, choices=[1, 3, 5, 10])
    parser.add_argument('--num_clips', default=1, type=int)

    parser.add_argument('--cpu', action='store_true', help='using cpu only')

    # for distributed learning
    parser.add_argument('--sync-bn', action='store_true',
                        help='sync BN across GPUs')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    return parser
