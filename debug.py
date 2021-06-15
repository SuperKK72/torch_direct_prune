from __future__ import print_function
import os
os.environ['GLOG_minloglevel'] = '3'
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import caffe
from tool import *

global args, parser
args, parser = parse_args()

# prototxt_path = "./models/vgg16/deploy.prototxt"
# caffemodel_path = "./models/vgg16/deploy.caffemodel"

net_name = "vgg16"
prototxt_path = "./models/vgg16/deploy.prototxt"
caffemodel_path = "./models/vgg16/deploy.caffemodel"
valid_dataset_path = "./valid"
synset_path = "./synset.txt"
label_path = "./valid.txt"
pruning_step = 48
pruning_ratio = 16

result_file = "{}_{}_{}".format(net_name,pruning_ratio,pruning_step)
result_path = "./{}.txt".format(result_file)

#analyse args
classes = np.loadtxt(synset_path, str, delimiter='\t')
images = []
labels = []
# labels_temp = []
dataset = open(label_path,'r')
# print(label_path)
for line in dataset:
    #print(line)
    line_list = line.strip().split(' ')
    #print(line_list)
    #images.append(os.path.join(valid_dataset_path, line_list[0]))
    labels.append(line_list[-1])
    images.append(os.path.join(valid_dataset_path, line_list[0]))
dataset.close()
# print(len(labels))

# image_num = len(images)
image_num = 1000
layer_num = 0
layer_names = []
caffe.set_mode_gpu()
net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)
wts = net.params
layers = net.layers
blobs = net.blobs

'''get original flops:'''
conv_layer_names = []
print("Searching all conv layers...")
flops = 0
f_cnt = 0
flops_queue = []
for name, _ in net.params.items():
    wt_shape = net.params[name][0].data.shape
    print(type(wt_shape))
    ft_shape = blobs[name].data.shape
    if len(wt_shape)==4:
        conv_layer_names.append(name)
        c_in = wt_shape[1]
        k_h = wt_shape[2]
        k_w = wt_shape[3]
        c_out = ft_shape[1]
        h_out = ft_shape[2]
        w_out = ft_shape[3]
        l_flops = c_out * h_out * w_out * c_in * k_h * k_w
        flops_queue.append(l_flops)
        flops += l_flops
        print("layer{:3d}: {:10s}\twt shape: {:20s}\tft shape: {:20s}\tflops: {}".format(f_cnt, name, wt_shape, ft_shape, l_flops))
        f_cnt += 1
    elif len(wt_shape)==2:
        l_flops = wt_shape[0] * wt_shape[1]
        flops += l_flops
        f_cnt += 1
        print("layer{:3d}: {:10s}\twt shape: {:20s}\tft shape: {:20s}\tflops: {}".format(f_cnt, name, wt_shape, ft_shape, l_flops))

print("total flops: {}".format(flops))
print("------------------------------------------------------")
exit()
# for layer_name, blob in net.blobs.iteritems():
#     print(layer_name + '\t' + str(blob.data.shape))

cnt = 0
print("Searching layers can be pruned...")
for name, _ in net.params.items():
    shape = net.params[name][0].data.shape
    if (len(shape) == 4 and shape[1] >= pruning_step):
        layer_names.append(name)
        cnt += 1
layer_num = len(layer_names)
print("All {} layers...".format(layer_num))
for l_idx in range(layer_num):
    print("layer{:<3d}: {:<15s}\tshape: {}".format(l_idx, layer_names[l_idx], net.params[layer_names[l_idx]][0].data.shape))
print("------------------------------------------------------")
