### TSN model - Caffe2pytorch 

#### This project transfers tsn-caffe model to tan-pytorch model. 

##### The model dir saves my transferred model. I test them on UCF101 and HMDB51 dataset, and get comparable results with the paper.

##### This project has three steps

- first get .hdf5file from caffemodel by export_to_hdf5.py.
- then use the .hdf5 file to get torch model by googlenet.lua. (Because the kinetics caffe model modified the layer name, there is small change in googlenet_kinetics.lua)
- finally transfer torch model to pytorch model by convert_torch.py.

##### Something to be noticed:

- You also need to modify test_videos.py because of the problem of state_dict, please see details in test_videos.py



