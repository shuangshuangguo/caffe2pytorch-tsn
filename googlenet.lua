require 'nn'
require 'inn'
require 'hdf5'
require 'nn'
function InceptionModule(name, inplane, outplane_a1x1, outplane_b3x3_reduce, outplane_b3x3, outplanedouble_3x3_reduce, outplanedouble_3x3_1, outplanedouble_3x3_2, outplane_pool_proj)
  if type(outplane_a1x1) == 'number' then
    a = nn.Sequential()
    a1x1 = nn.SpatialConvolution(inplane, outplane_a1x1, 1, 1, 1, 1, 0, 0)
    a1x1.name = name .. '/1x1'
    bn1x1 = nn.SpatialBatchNormalization(outplane_a1x1)
    bn1x1.name = name .. '/1x1_bn'
    a:add(a1x1):add(bn1x1)
    a:add(nn.ReLU(true))
  end

  b = nn.Sequential()
  b3x3_reduce = nn.SpatialConvolution(inplane, outplane_b3x3_reduce, 1, 1, 1, 1, 0, 0)
  b3x3_reduce.name = name .. '/3x3_reduce'
  bn3x3_reduce = nn.SpatialBatchNormalization(outplane_b3x3_reduce)
  bn3x3_reduce.name = name .. '/3x3_reduce_bn'
  b:add(b3x3_reduce):add(bn3x3_reduce)
  b:add(nn.ReLU(true))
  b3x3 = nn.SpatialConvolution(outplane_b3x3_reduce, outplane_b3x3, 3, 3, 1, 1, 1, 1)
  b3x3.name = name .. '/3x3'
  bn3x3 = nn.SpatialBatchNormalization(outplane_b3x3)
  bn3x3.name = name .. '/3x3_bn'
  b:add(b3x3):add(bn3x3)
  b:add(nn.ReLU(true))

  c = nn.Sequential()
  double_3x3_reduce = nn.SpatialConvolution(inplane, outplanedouble_3x3_reduce, 1, 1, 1, 1, 0, 0)
  double_3x3_reduce.name = name .. '/double_3x3_reduce'
  bndouble_3x3_reduce = nn.SpatialBatchNormalization(outplanedouble_3x3_reduce)
  bndouble_3x3_reduce.name = name .. '/double_3x3_reduce_bn'
  c:add(double_3x3_reduce):add(bndouble_3x3_reduce)
  c:add(nn.ReLU(true))
  double_3x3_1 = nn.SpatialConvolution(outplanedouble_3x3_reduce, outplanedouble_3x3_1, 3, 3, 1, 1, 1, 1)
  double_3x3_1.name = name .. '/double_3x3_1'
  bndouble_3x3_1 = nn.SpatialBatchNormalization(outplanedouble_3x3_1)
  bndouble_3x3_1.name = name .. '/double_3x3_1_bn'
  c:add(double_3x3_1):add(bndouble_3x3_1)
  c:add(nn.ReLU(true))

  double_3x3_2 = nn.SpatialConvolution(outplanedouble_3x3_1, outplanedouble_3x3_2, 3, 3, 1, 1, 1, 1)
  double_3x3_2.name = name .. '/double_3x3_2'
  bndouble_3x3_2 = nn.SpatialBatchNormalization(outplanedouble_3x3_2)
  bndouble_3x3_2.name = name .. '/double_3x3_2_bn'
  c:add(double_3x3_2):add(bndouble_3x3_2)
  c:add(nn.ReLU(true))


  d = nn.Sequential()
  if name == 'inception_4e' or name == 'inception_3c' then
    d:add(nn.SpatialMaxPooling(3, 3, 2, 2))
  elseif name == 'inception_5b' then
      d:add(nn.SpatialMaxPooling(3, 3, 1, 1, 1, 1))
  else
      d:add(nn.SpatialAveragePooling(3, 3, 1, 1, 1, 1))
  end
  if name ~= 'inception_4e' and name ~= 'inception_3c' then
    d_pool_proj = nn.SpatialConvolution(inplane, outplane_pool_proj, 1, 1, 1, 1, 0, 0)
    d_pool_proj.name = name .. '/pool_proj'
    bnd_pool_proj = nn.SpatialBatchNormalization(outplane_pool_proj)
    bnd_pool_proj.name = name .. '/pool_proj_bn'
    d:add(d_pool_proj):add(bnd_pool_proj)
    d:add(nn.ReLU(true))
  end

  if type(outplane_a1x1) == 'number' then
    net = nn.Sequential():add(nn.ConcatTable():add(a):add(b):add(c):add(d)):add(nn.JoinTable(2))
  elseif type(outplane_a1x1) == 'string' then
    net = nn.Sequential():add(nn.ConcatTable():add(b):add(c):add(d)):add(nn.JoinTable(2))
  else
    print('unexpected type of outplane_a1x1', type(outplane_a1x1))
  end
  return net
end

for k,v in ipairs(arg) do
  if k==1 then dataset = v end
  if k==2 then stream = v end
  if k==3 then h5 = v end
  if k==4 then t7 = v end
end

local model = nn.Sequential()
if stream == 'rgb'then
  conv1_7x7_s2 = nn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3)
else
  conv1_7x7_s2 = nn.SpatialConvolution(10, 64, 7, 7, 2, 2, 3, 3)
end
conv1_7x7_s2.name = 'conv1/7x7_s2'
bnconv1_7x7_s2 = nn.SpatialBatchNormalization(64)
bnconv1_7x7_s2.name = 'conv1/7x7_s2_bn'
model:add(conv1_7x7_s2):add(bnconv1_7x7_s2)
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(3, 3, 2, 2):ceil())
--model:add(inn.SpatialCrossResponseNormalization(5, 0.0001, 0.75, 1))

local conv_3x3_reduce = nn.SpatialConvolution(64, 64, 1, 1, 1, 1, 0, 0)
conv_3x3_reduce.name = 'conv2/3x3_reduce'
bnconv_3x3_reduce = nn.SpatialBatchNormalization(64)
bnconv_3x3_reduce.name = 'conv2/3x3_reduce_bn'
model:add(conv_3x3_reduce):add(bnconv_3x3_reduce)
model:add(nn.ReLU(true))
local conv_3x3 = nn.SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1)
conv_3x3.name = 'conv2/3x3'
bnconv_3x3 = nn.SpatialBatchNormalization(192)
bnconv_3x3.name = 'conv2/3x3_bn'
model:add(conv_3x3):add(bnconv_3x3)
model:add(nn.ReLU(true))
--model:add(inn.SpatialCrossResponseNormalization(5, 0.0001, 0.75, 1))
model:add(nn.SpatialMaxPooling(3, 3, 2, 2):ceil())

model:add(InceptionModule('inception_3a', 192, 64, 64, 64, 64, 96, 96, 32))
model:add(InceptionModule('inception_3b', 256, 64, 64, 96, 64, 96, 96, 64))
model:add(InceptionModule('inception_3c', 320, '', 128, 160, 64, 96, 96, ''))

model:add(InceptionModule('inception_4a', 576, 224, 64, 96, 96, 128, 128, 128))
model:add(InceptionModule('inception_4b', 576, 192, 96, 128, 96, 128, 128, 128))
model:add(InceptionModule('inception_4c', 576, 160, 128, 160, 128, 160, 160, 128))
model:add(InceptionModule('inception_4d', 608, 96, 128, 192, 160, 192, 192, 128))
model:add(InceptionModule('inception_4e', 608, '', 128, 192, 192, 256, 256, ''))

model:add(InceptionModule('inception_5a', 1056, 352, 192, 320, 160, 224, 224, 128))
model:add(InceptionModule('inception_5b', 1024, 352, 192, 320, 192, 224, 224, 128))

model:add(nn.SpatialAveragePooling(7, 7, 1, 1))
drop = nn.Dropout(0.8)
drop.name = 'dropout'
model:add(drop)
model:add(nn.View(-1, 1024))

if dataset=='ucf101' then
  classifier = nn.Linear(1024, 101)
elseif dataset == 'hmdb51' then
  classifier = nn.Linear(1024, 51)
elseif dataset == 'kinetics' then
  classifier = nn.Linear(1024, 400)
else
  print('unknown dataset')
end

classifier.name = 'fc-action'
model:add(classifier)


local paramsFile = hdf5.open(h5, 'r')
local moduleQueue = { model }
local touchedLayers = { }
for k1, v1 in ipairs(moduleQueue) do
  if v1.modules then
    for k2, v2 in ipairs(v1.modules) do
      table.insert(moduleQueue, v2)
    end
  end

  if v1.name then
    touchedLayers[v1.name] = true
    local layer = paramsFile:read(v1.name):all()
    if layer['000'] then
      v1.weight:copy(layer['000'])
    else
      print(v1.name .. ' has no weight')
    end
    if layer['001'] then
      v1.bias:copy(layer['001'])
    else
      print(v1.name .. ' has no bias')
    end
    if layer['002'] then
      v1.running_mean:copy(layer['002'])
    end
    if layer['003'] then
      v1.running_var:copy(layer['003'])
    end
  end
end

paramsFile:close()

torch.save(t7, model)
