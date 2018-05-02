import sys
sys.path.append('/path/temporal-segment-networks/lib/caffe-action')
from caffe.proto import caffe_pb2

import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help='caffemodel need to transfer')
parser.add_argument("--hdf5", type=str, help='hdf5 file')
args = parser.parse_args()

dirs = '/path/temporal-segment-networks/models/'
files =dirs + args.model
net_param = caffe_pb2.NetParameter()
with open(files, 'r') as f:
  net_param.ParseFromString(f.read())

output_file = h5py.File(args.hdf5, 'w')

for layer in net_param.layer:
    group = output_file.create_group(layer.name)
    print layer.name
    for pos, blob in enumerate(layer.blobs):
        dims = []
        for dim in blob.shape.dim:
            dims.append(dim)
        if len(dims)==1:
            data = np.array(blob.data).reshape(dims[0])
        if len(dims)==2:
            data = np.array(blob.data).reshape(dims[0], dims[1])
        if len(dims)==3:
            data = np.array(blob.data).reshape(dims[0], dims[1], dims[2])
        if len(dims)==4:
            data = np.array(blob.data).reshape(dims[0], dims[1], dims[2], dims[3])
        dataset = group.create_dataset('%03d' % pos, data=data)

output_file.close()
