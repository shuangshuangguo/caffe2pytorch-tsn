from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.serialization import load_lua

import numpy as np
import os
import math
from functools import reduce

parser = argparse.ArgumentParser()
parser.add_argument("--t7", type=str, help='torchmodel need to transfer')
parser.add_argument("--pth", type=str, help='pytorch model need to save')
args = parser.parse_args()

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        # result is Variables list [Variable1, Variable2, ...]
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        # result is a Variable
        return reduce(self.lambda_func,self.forward_prepare(input))


def copy_param(m,n):
    if m.weight is not None: n.weight.data.copy_(m.weight)
    if m.bias is not None: n.bias.data.copy_(m.bias)
    if hasattr(n,'running_mean'): n.running_mean.copy_(m.running_mean)
    if hasattr(n,'running_var'): n.running_var.copy_(m.running_var)

def add_submodule(names, seq, *args):
    for n in args:
        seq.add_module(names,n)

def lua_recursive_model(module,seq):
    for m in module.modules:
        name = type(m).__name__
        real = m
        if name == 'TorchObject':
            name = m._typename.replace('cudnn.','')
            m = m._obj

        if name == 'SpatialConvolution' or name == 'nn.SpatialConvolution':
            if not hasattr(m,'groups') or m.groups is None: m.groups=1
            n = nn.Conv2d(m.nInputPlane,m.nOutputPlane,(m.kW,m.kH),(m.dW,m.dH),(m.padW,m.padH),1,m.groups,bias=(m.bias is not None))
            copy_param(m,n)
            names = m.name
            names = 'base_model.' + names.replace('/', '_');
            add_submodule(names, seq, n)
        elif name == 'SpatialBatchNormalization':
            n = nn.BatchNorm2d(m.running_mean.size(0), m.eps, m.momentum, m.affine)
            copy_param(m,n)
            names = m.name
            names = 'base_model.' + names.replace('/', '_')
            add_submodule(names,seq,n)
        elif name == 'ReLU':
            n = nn.ReLU()
            add_submodule(str(len(seq._modules)),seq,n)
        elif name == 'SpatialMaxPooling':
            n = nn.MaxPool2d((m.kW,m.kH),(m.dW,m.dH),(m.padW,m.padH),ceil_mode=m.ceil_mode)
            add_submodule(str(len(seq._modules)),seq,n)
        elif name == 'SpatialAveragePooling':
            n = nn.AvgPool2d((m.kW,m.kH),(m.dW,m.dH),(m.padW,m.padH),ceil_mode=m.ceil_mode)
            add_submodule(str(len(seq._modules)),seq,n)
        elif name == 'View':
            n = Lambda(lambda x: x.view(x.size(0),-1))
            add_submodule(str(len(seq._modules)),seq,n)
        elif name == 'Reshape':
            n = Lambda(lambda x: x.view(x.size(0),-1))
            add_submodule(str(len(seq._modules)),seq,n)
        elif name == 'Linear':
            # Linear in pytorch only accept 2D input
            n1 = Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x )
            n2 = nn.Linear(m.weight.size(1),m.weight.size(0),bias=(m.bias is not None))
            copy_param(m,n2)
            n = nn.Sequential(n1,n2)
            names = m.name
            names = 'base_model.' + names.replace('/', '_');
            add_submodule(names,seq,n)
        elif name == 'Dropout':
            m.inplace = False
            n = nn.Dropout(m.p)
            add_submodule(str(len(seq._modules)),seq,n)
        elif name == 'SoftMax':
            n = nn.Softmax()
            add_submodule(str(len(seq._modules)),seq,n)
        elif name == 'Sequential':
            n = nn.Sequential()
            lua_recursive_model(m,n)
            add_submodule(str(len(seq._modules)),seq,n)
        elif name == 'ConcatTable': # output is list
            n = LambdaMap(lambda x: x)
            lua_recursive_model(m,n)
            add_submodule(str(len(seq._modules)),seq,n)
        elif name == 'CAddTable': # input is list
            n = LambdaReduce(lambda x,y: x+y)
            add_submodule(str(len(seq._modules)),seq,n)
        elif name == 'JoinTable':
            dim = m.dimension
            n = LambdaReduce(lambda x,y,dim=dim: torch.cat((x,y),dim))
            #lua_recursive_model(m,n)
            add_submodule(str(len(seq._modules)),seq,n)
        elif name == 'TorchObject':
            print('Not Implement',name,real._typename)
        else:
            print('Not Implement',name)

def torch_to_pytorch(t7_filename,outputname=None):
    model = load_lua(t7_filename,unknown_classes=True)
    if type(model).__name__=='hashable_uniq_dict': model=model.model
    model.gradInput = None
    n = nn.Sequential()
    lua_recursive_model(model,n)
    torch.save(n.state_dict(),outputname)

if __name__ == '__main__':
    model = args.t7
    output = args.pth
    torch_to_pytorch(model, output)
