#!/usr/bin/env bash

dataset=$1
stream=$2
# example: bash caffe2pytorch.sh ucf101 rgb

echo "transfer caffemodel to pytorchmodel for TSN:"
echo "first get .hdf5 file:"
if python export_to_hdf5.py --model ${dataset}_split_1_tsn_${stream}_reference_bn_inception.caffemodel --hdf5 model/${dataset}_1_${stream}.h5; then
    echo "then get torchmodel:"
    th googlenet.lua ${dataset} ${stream} model/${dataset}_1_${stream}.h5 model/${dataset}_1_${stream}.t7
    if th googlenet.lua ${dataset} ${stream} model/${dataset}_1_${stream}.h5 model/${dataset}_1_${stream}.t7; then
        echo "finally transfer torchmodel to pytorchmodel"
        python convert_torch.py --t7 model/${dataset}_1_${stream}.t7 --pth model/${dataset}_1_${stream}.pth
    else
        exit 1
    fi
else
    exit 1
fi


