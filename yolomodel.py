from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
import onnx
from onnx import numpy_helper

class MaxPoolSameSize(nn.Module):
    '''
      specific for that stride=1, kernel_size=3, padding layer in tinyyolo
    '''
    def __init__(self, kernel_size=2, stride=1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = F.pad(x, (0,1,0,1), mode='replicate')
        x = self.maxpool(x)
        return x
    
class Block(nn.Module):
    '''
      define a block which has
      1. conv
      2. batchnorm: no running mean
      3. leakyrelu: alfa=0.1
      4. maxpool: 
          normal: regular maxpool
          same: Maxpoolsamesize
          none: no maxpool
    '''
    def __init__(self,
                 in_channel,
                 out_channel,
                 maxpool=None,
                 conv_bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size=3,
                              stride=1,
                              padding='same',
                              bias=conv_bias)
        self.norm = nn.BatchNorm2d(out_channel, track_running_stats=True, momentum=0.0)
        self.activation = nn.LeakyReLU(0.1)
        if maxpool == 'normal':
            self.maxpool = nn.MaxPool2d(kernel_size=2,
                                        stride=2)
        elif maxpool == 'same':
            self.maxpool = MaxPoolSameSize()
        else:
            self.maxpool = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x

class MyTinyYolo(nn.Module):
    def __init__(self, num_class=20, num_bbox=5, num_grid=13):
        '''
          set up the tiny-yolo model according to the onnx equivalent
          the output class is 20, with 5 bounding box, so the final out put is (20+5)*5=125
          the final output has a grid size of 13, so the input x has a size of 416
        '''
        super().__init__()
        self.model = nn.Sequential(
            Block(3, 16, 'normal'),
            Block(16, 32, 'normal'),
            Block(32, 64, 'normal'),
            Block(64, 128, 'normal'),
            Block(128, 256, 'normal'),
            Block(256, 512, 'same'),
            Block(512, 1024, None),
            Block(1024, 1024, None),
            nn.Conv2d(1024, 125, kernel_size=1, bias=True)
        )
        # from https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny-voc.cfg
        self.anchors = [(1.08,1.19), (3.42,4.41), (6.63,11.38), (9.42,5.11), (16.62,10.52)]
        # from https://github.com/pjreddie/darknet/blob/master/data/voc.names
        self.labels = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
                       'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

    def load(self, onnx_model='tinyyolov2-7.onnx'):
        # load onnx_model and the weight
        onnx_model = onnx.load(onnx_model)
        onnx_weight = {}
        for initializer in onnx_model.graph.initializer:
            W = numpy_helper.to_array(initializer)
            onnx_weight[initializer.name] = W
            
        # convert onnx_weight to state_dict
        onnx_state_dict = {}
        for name, param in onnx_weight.items():
            if name.startswith('convolution'):
                layer_type = 'conv'
                stage, part = name.split('_')
                stage = stage[-1]
                if stage == 'n':
                    stage = '0'
                if part == 'W':
                    part = 'weight'
                else:
                    part = 'bias'
            elif name.startswith('Batch'):
                layer_type = 'norm'
                stage = name[-1]
                if stage.isnumeric() == False:
                    stage = '0'
                part = name.split('_')[-1]
                if part.startswith('scale'):
                    part = 'weight'
                elif part.startswith('B'):
                    part = 'bias'
                elif part.startswith('mean'):
                    part = 'running_mean'
                elif part.startswith('var'):
                    part = 'running_var'
                else:
                    print('missing:', name)
            else:
                continue
            if stage == '8':
                onnx_state_dict[f'model.{stage}.{part}'] = torch.tensor(param)
            else:
                onnx_state_dict[f'model.{stage}.{layer_type}.{part}'] = torch.tensor(param)

        # load onnx_state_dict
        torch_state_dict = self.state_dict()
        torch_state_dict.update(onnx_state_dict)
        print(torch_state_dict.keys())
        print(onnx_state_dict.keys())
        print(self.load_state_dict(torch_state_dict))

    def forward(self, x):
        x = self.model(x)
        return x