# Action Recognition Study

This repository contains a general implementation of 6 representative 2D and 3D approaches for action recognition including I3D [1], ResNet3D [2], S3D [3], R(2+1)D [4], TSN [5] and TAM [6]. 

```
@incollection{
    fan2019blvnet,
    title={{More Is Less: Learning Efficient Video Representations by Temporal Aggregation Modules}},
    author={Quanfu Fan and Chun-Fu (Ricarhd) Chen and Hilde Kuehne and Marco Pistoia and David Cox},
    booktitle={Advances in Neural Information Processing Systems 33},
    year={2019}
}
```

## Requirements

```
pip3 install torch
pip3 install torchvision
pip3 install tqdm pyarrow lmdb tensorboard_logger
pip3 install git+https://github.com/chunfuchen/pytorch-summary
```

## Data Preparation
We provide two scripts in the folder `tools` for prepare input data for model training. The scripts sample an image sequence from a video and then resize each image to have its shorter side to be `256` while keeping the aspect ratio of the image.
You may need to set up `folder_root` accordingly to assure the extraction works correctly.
 
## training script

```
python3 train.py --datadir /path/to/folder --threed_data \
--dataset st2stv1 --frames_per_group 1 --groups 8 \
--logdir snapshots/ --lr 0.01 --backbone_net i3d_resnet -b 64 -j 36
```

## training script, distributed version on a single node

Not tested for multiple nodes yet.
The distributed version supports synchorize the running mean and var in the batchnorm layer

```
python3 train_dist.py --datadir /path/to/folder --threed_data \
--dataset st2stv1 --frames_per_group 1 --groups 8 \
--logdir snapshots/ --lr 0.01 --backbone_net i3d_resnet -b 64 -j 36 --sync-bn --multiprocessing-distributed
```

## training script, distributed version on multi-node

It works but not validate the accuracy yet.
The distributed version supports synchorize the running mean and var in the batchnorm layer

Launch the following command on each machine individually, E.g. two nodes:
The first node:
```
python3 train_dist.py --datadir /path/to/folder --threed_data \
--dataset st2stv1 --frames_per_group 1 --groups 8 \
--logdir snapshots/ --lr 0.01 --backbone_net i3d_resnet -b 64 -j 36 --sync-bn --multiprocessing-distributed \
--world-size 2 --rank 0 --dist-url tcp://URL:PORT
```

The second node (put the same `tcp://URL:PORT` to the first node, since the first node will be used as parameter server):
```
python3 train_dist.py --datadir /path/to/folder --threed_data \
--dataset st2stv1 --frames_per_group 1 --groups 8 \
--logdir snapshots/ --lr 0.01 --backbone_net i3d_resnet -b 64 -j 36 --sync-bn --multiprocessing-distributed \
--world-size 2 --rank 1 --dist-url tcp://URL:PORT
```

## training with lmdb
Everything is the same to above examples, you just need to add one more flag, `--use_lmdb`
```
python3 train.py --datadir /path/to/folder --threed_data \
--dataset st2stv1 --frames_per_group 1 --groups 8 \
--logdir snapshots/ --lr 0.01 --backbone_net i3d_resnet -b 64 -j 36 --use_lmdb
```

## Testing
python3 test.py --datadir /path/to/folder --threed_data \
--dataset st2stv1 --frames_per_group 1 --groups 8 \
--pretrained /path/to/model.pth.tar \
--logdir snapshots/ --backbone_net i3d_resnet -b 12 -j 36 \
--num_clips 3 --num_crops 3 \ 
--evaluate --num_clips 3 --num_crops 3 
