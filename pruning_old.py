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

prototxt_path = "./models/mobilenet_v1/deploy.prototxt"
caffemodel_path = "./models/mobilenet_v1/deploy.caffemodel"

valid_dataset_path = "./valid"
synset_path = "./synset.txt"
label_path = "./valid.txt"

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
dataset.close()
print(len(labels))
val_dataset_file_path = "./valid.txt"
for root, dirs, files in os.walk(valid_dataset_path):
    # print(files)
    cnt = len(files)
    print(cnt)
    for i in range(cnt-1):
        images.append(os.path.join(valid_dataset_path, files[i]))


image_num = len(images)
# image_num = 100
caffe.set_mode_cpu()
net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

#for vgg16 pruning
wts = net.params
layer_num = 0
layer_names = []

is_pruning = True
if(is_pruning):
    cnt = 0
    for name, _ in net.params.items():
        if(len(net.params[name][0].data.shape) == 4):
            layer_names.append(name)
            print("{}: name: {}\tshape: {}\n".format(cnt, name, net.params[name][0].data.shape))
            cnt+=1
    layer_num = len(layer_names)
    print("All {} layers...".format(layer_num))
    print("------------------------------------------------------")

    for i in range(layer_num):
        print("pruning layer {}...".format(i))
        if(i==0):
            continue
        curr_wt = wts[layer_names[i]][0].data
        print("wt size: {}\twt shape: {}\n".format(curr_wt.size, curr_wt.shape))
        k_num = curr_wt.shape[0]
        c = curr_wt.shape[1]
        h = curr_wt.shape[2]
        w = curr_wt.shape[3]
        pruning_step = 20
        pruning_ratio = 4
        #note c can divide by 4
        pruning_num = c//pruning_step
        s = pruning_step
        r = pruning_ratio
        t = pruning_num


        for i in range(k_num):
            k = curr_wt[i]
            # print(k)
            for h_idx in range(h):
                for w_idx in range(w):
                    # min_val = min(k[:, h_idx, w_idx])
                    # min_idx = np.argmin(abs(k[:, h_idx, w_idx]))
                    # print("max_val: {}\tmax_idx: {}\n".format(max_val, max_idx))
                    # print("min_val: {}\tmin_idx: {}\n".format(min_val, min_idx))
                    # k[min_idx, h_idx, w_idx] = 0
                    for c_idx in range(t):
                        # min_idx = np.argmin(abs(k[c_idx*s:(c_idx+1)*s, h_idx, w_idx]))
                        # print("c_idx: {}\tmin_idx: {}".format(c_idx, c_idx*s+min_idx))
                        sorted_idx = np.argsort(abs(k[c_idx*s:(c_idx+1)*s, h_idx, w_idx]))
                        for r_idx in range(r):
                            k[c_idx*s+sorted_idx[r_idx], h_idx, w_idx] = 0

            # print(k)
        print("--------------------------------------------------------")


top1_cnt = 0
top5_cnt = 0
log_path = "./log/log.txt"
image_num = 1000
log_file = open(log_path, 'w')
for i in range(image_num):
    print("eval image {}/{}......".format(i, image_num))
    log_file.write("eval image {}/{}......\n".format(i, image_num))
    log_file.write("ground truth: {}\n".format(labels[i]))
    prob, idx = eval(net, images[i])
    if int(labels[i]) == int(idx[0]):
        top1_cnt = top1_cnt+1
    if int(labels[i]) in idx:
        top5_cnt = top5_cnt+1
    for i in range(5):
        label = idx[i]
        #print('label: %d confidence: %.2f --- %s' % (label, prob[label], classes[label]))
        log_file.write('label: %d confidence: %.6f --- %s\n' % (label, prob[label], classes[label]))
    #print("---------------------------------")
    log_file.write("---------------------------------\n")
print("top1_cnt:{}\ttop5_cnt:{}\tall:{}\n".format(top1_cnt,top5_cnt,image_num))
top1_acc = 1.0 * top1_cnt / image_num
top5_acc = 1.0 * top5_cnt / image_num
print("top1 accuracy: {}    top5 accuracy: {}".format(top1_acc, top5_acc))
log_file.write("top1 accuracy: {}    top5 accuracy: {}\n".format(top1_acc, top5_acc))
log_file.close()





