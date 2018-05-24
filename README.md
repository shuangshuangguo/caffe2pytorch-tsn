### TSN model - Caffe2pytorch 

#### This project transfers tsn-caffe model to tan-pytorch model. 

##### The model dir saves my transferred model. I test them on UCF101 and HMDB51 dataset, and get comparrable results with the paper.

##### This project has three steps

- first get .hdf5file from caffemodel in python
- then use the .hdf5 file to get torch model in lua
- finally transfer torch model to pytorch model in python.

##### Something to be noticed:

- You also need to modify test_videos.py because of the problem of state_dict, please see details in test_videos.py



