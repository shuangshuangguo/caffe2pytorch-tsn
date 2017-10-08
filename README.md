# caffe2pytorch-tsn

This scipt first gets .hdf5file from caffemodel in python, 
then use this .hdf5 file to get torch model in lua, 
finally we transfer torch model to pytorch model in python.

(Maybe you can use 'load_lua' and 'torch.save' directly in pytorch, but you can find that there will be error when you use this pytorchmodel on test_videos.py.)

To be honest, this script is so complicated. I'm looking forward to see yjxiong's transferring work.
