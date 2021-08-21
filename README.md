# FedSkel

This repo is the implementation of **FedSkel: Efficient Federated Learning on Heterogeneous Systems with Skeleton Gradients Update** [CIKM'21].

## Paper
FedSkel: Efficient Federated Learning on Heterogeneous Systems with Skeleton Gradients Update

Doi: 10.1145/3459637.3482107

## Accuracy

We verified that FedSkel will not affect FL accuracy with PyTorch Framework. The core codes are in `./acc-pytorch`. We adapted the FL framework of [LG-FedAvg](https://github.com/pliang279/LG-FedAvg) for FedSkel.

## Speed Up

We measure the speedups of FedSkel on mobile devices with Caffe framework. The core codes are in `./speedup-caffe`. We modified CONV layers of Caffe to enable gradients' pruning in the *SetSkel* and *UpdateSkel* processes during training.
