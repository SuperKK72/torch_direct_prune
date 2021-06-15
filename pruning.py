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
pruning_step = 3
pruning_ratio = 1

result_file = "{}_{}_{}".format(net_name,pruning_ratio,pruning_step)
result_path = "./{}.txt".format(result_file)
fresult = open(result_path, 'w')
fresult.write("net name: {}\n".format(net_name))
fresult.write("prune mode: {}/{}\n".format(pruning_ratio,pruning_step))
fresult.write("------------------------------------------------------\n")

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
fresult.write("Searching all conv layers...\n")
flops = 0
f_cnt = 0
flops_queue = []
param = 0
l_param_queue = []
for name, _ in net.params.items():
    wt_shape = net.params[name][0].data.shape
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
        l_param = wt_shape[0] * wt_shape[1] * wt_shape[2] * wt_shape[3]
        l_param_queue.append(l_param)
        param += l_param
        f_cnt += 1
    elif len(wt_shape)==2:
        l_flops = wt_shape[0] * wt_shape[1]
        flops_queue.append(l_flops)
        flops += l_flops
        l_param = wt_shape[0] * wt_shape[1]
        l_param_queue.append(l_param)
        param += l_param
        f_cnt += 1
    print("layer{:<3d}: {:10s}\twt shape: {:20s}\tft shape: {:20s}\tflops: {:<12d}\tparam: {}".format(f_cnt, name, wt_shape, ft_shape, l_flops, l_param))
print("total flops: {}".format(flops))
print("total param: {}".format(param))
print("------------------------------------------------------")
fresult.write("total flops: {}\n".format(flops))
fresult.write("total param: {}\n".format(param))
fresult.write("------------------------------------------------------\n")
# fresult.close()
# exit()

# for layer_name, blob in net.blobs.iteritems():
#     print(layer_name + '\t' + str(blob.data.shape))

cnt = 0
print("Searching layers can be pruned...")
fresult.write("Searching layers can be pruned...\n")
for name, _ in net.params.items():
    shape = net.params[name][0].data.shape
    if (len(shape) == 4 and shape[1] >= pruning_step):
        layer_names.append(name)
        cnt += 1
    if (len(shape) == 2 and shape[1] >= pruning_step):
        layer_names.append(name)
        cnt += 1
layer_num = len(layer_names)
print("All {} layers...".format(layer_num))
fresult.write("All {} layers...\n".format(layer_num))
for l_idx in range(layer_num):
    print("layer{:<3d}: {:<15s}\tshape: {}".format(l_idx, layer_names[l_idx], net.params[layer_names[l_idx]][0].data.shape))
    fresult.write("{:<3d}: name: {:<15s}\tshape: {}\n".format(l_idx, layer_names[l_idx], net.params[layer_names[l_idx]][0].data.shape))
print("------------------------------------------------------")
fresult.write("------------------------------------------------------\n")




top1_acc_queue = []
for l_idx in range(layer_num):
    net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)
    wts = net.params
    print("pruning layer {}...".format(l_idx))
    curr_wt = wts[layer_names[l_idx]][0].data
    print("wt size: {}\twt shape: {}".format(curr_wt.size, curr_wt.shape))
    if(len(curr_wt.shape)==4):
        k_num = curr_wt.shape[0]
        c = curr_wt.shape[1]
        h = curr_wt.shape[2]
        w = curr_wt.shape[3]
        #note c can divide by 4
        pruning_num = c//pruning_step
        s = pruning_step
        r = pruning_ratio
        t = pruning_num
        for i in range(k_num):
            k = curr_wt[i]
            for h_idx in range(h):
                for w_idx in range(w):
                    for c_idx in range(t):
                        sorted_idx = np.argsort(abs(k[c_idx*s:(c_idx+1)*s, h_idx, w_idx]))
                        for r_idx in range(r):
                            k[c_idx*s+sorted_idx[r_idx], h_idx, w_idx] = 0
    elif (len(curr_wt.shape) == 2):
        k_num = curr_wt.shape[0]
        c = curr_wt.shape[1]
        # note c can divide by 4
        pruning_num = c // pruning_step
        s = pruning_step
        r = pruning_ratio
        t = pruning_num
        for i in range(k_num):
            k = curr_wt[i]
            for c_idx in range(t):
                sorted_idx = np.argsort(abs(k[c_idx * s:(c_idx + 1) * s]))
                for r_idx in range(r):
                    k[c_idx * s + sorted_idx[r_idx]] = 0
    top1_cnt = 0
    top5_cnt = 0
    for i in range(image_num):
        if i%100==0:
            print("eval image {}/{}......".format(i, image_num))
        prob, idx = eval(net, images[i])
        if int(labels[i]) == int(idx[0]):
            top1_cnt = top1_cnt + 1
        if int(labels[i]) in idx:
            top5_cnt = top5_cnt + 1
        for i in range(5):
            label = idx[i]
    print("top1_cnt:{}\ttop5_cnt:{}\tall:{}".format(top1_cnt, top5_cnt, image_num))
    top1_acc = 1.0 * top1_cnt / image_num
    top5_acc = 1.0 * top5_cnt / image_num
    print("top1 accuracy: {:.6f}\ttop5 accuracy: {:.6f}".format(top1_acc, top5_acc))
    top1_acc_queue.append(top1_acc)
    print("------------------------------------------------------")

# exit()
print("prune top1_acc queue...")
print(top1_acc_queue)
print("------------------------------------------------------")
fresult.write("prune top1_acc queue...\n")
fresult.write(str(top1_acc_queue)+"\n")
fresult.write("------------------------------------------------------\n")

top1_baseline = 0.61
print("original top1_acc: {}".format(top1_baseline))
print("------------------------------------------------------")
fresult.write("original top1_acc: {}\n".format(top1_baseline))
fresult.write("------------------------------------------------------\n")
prune_queue = []
prune_queue_pre = np.argsort(top1_acc_queue)
prune_queue_pre = prune_queue_pre[::-1]
print("prune queue pre...")
fresult.write("prune queue pre...\n")
print(prune_queue_pre)
fresult.write(str(prune_queue_pre)+"\n")
print("------------------------------------------------------")
fresult.write("------------------------------------------------------\n")

#find layers that l_top1_acc_decrease <= 0.5
for i in range(layer_num):
    err = top1_acc_queue[prune_queue_pre[i]] - top1_baseline
    if err >= -0.005:
        prune_queue.append(prune_queue_pre[i])
    # if top1_acc_queue[prune_queue_pre[i]] >= 0.7:
    #     prune_queue.append(prune_queue_pre[i])
print("prune queue...")
print(prune_queue)
print("------------------------------------------------------")
fresult.write("prune queue...\n")
fresult.write(str(prune_queue)+"\n")
fresult.write("------------------------------------------------------\n")

#find the prune_layer_queue that n_top1_acc_decrease >= 0.1 or 0.2
prune_queue_len = len(prune_queue)
net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)
wts = net.params
prune_top1_acc = []
pruned_layers = []
for p_idx in range(prune_queue_len):
    l_idx = prune_queue[p_idx]
    print("pruning layer {}...".format(l_idx))
    curr_wt = wts[layer_names[l_idx]][0].data
    # print("wt size: {}\twt shape: {}".format(curr_wt.size, curr_wt.shape))
    if(len(curr_wt.shape)==4):
        k_num = curr_wt.shape[0]
        c = curr_wt.shape[1]
        h = curr_wt.shape[2]
        w = curr_wt.shape[3]
        # note c can divide by 4
        pruning_num = c // pruning_step
        s = pruning_step
        r = pruning_ratio
        t = pruning_num
        for i in range(k_num):
            k = curr_wt[i]
            for h_idx in range(h):
                for w_idx in range(w):
                    for c_idx in range(t):
                        sorted_idx = np.argsort(abs(k[c_idx * s:(c_idx + 1) * s, h_idx, w_idx]))
                        for r_idx in range(r):
                            k[c_idx * s + sorted_idx[r_idx], h_idx, w_idx] = 0
    elif (len(curr_wt.shape) == 2):
        k_num = curr_wt.shape[0]
        c = curr_wt.shape[1]
        # note c can divide by 4
        pruning_num = c // pruning_step
        s = pruning_step
        r = pruning_ratio
        t = pruning_num
        for i in range(k_num):
            k = curr_wt[i]
            for c_idx in range(t):
                sorted_idx = np.argsort(abs(k[c_idx * s:(c_idx + 1) * s]))
                for r_idx in range(r):
                    k[c_idx * s + sorted_idx[r_idx]] = 0
    top1_cnt = 0
    top5_cnt = 0
    for i in range(image_num):
        if i % 20 == 0:
            '''print("eval image {}/{}......".format(i, image_num))'''
        prob, idx = eval(net, images[i])
        if int(labels[i]) == int(idx[0]):
            top1_cnt = top1_cnt + 1
        if int(labels[i]) in idx:
            top5_cnt = top5_cnt + 1
        for i in range(5):
            label = idx[i]
    top1_acc = 1.0 * top1_cnt / image_num
    top5_acc = 1.0 * top5_cnt / image_num
    if (top1_acc - top1_baseline) < -0.001:
        break
    pruned_layers.append(l_idx)
    print("top1_cnt:{}\ttop5_cnt:{}\tall:{}".format(top1_cnt, top5_cnt, image_num))
    print("top1 accuracy: {:.6f}\ttop5 accuracy: {:.6f}".format(top1_acc, top5_acc))

    prune_top1_acc.append(top1_acc)
    print("------------------------------------------------------")
print("prune top1_acc...")
print(prune_top1_acc)
print("------------------------------------------------------")
fresult.write("prune top1_acc...\n")
fresult.write(str(prune_top1_acc)+"\n")
fresult.write("------------------------------------------------------\n")

print("pruned layers...")
print(pruned_layers)
print("------------------------------------------------------")
fresult.write("pruned layers...\n")
fresult.write(str(pruned_layers)+"\n")
fresult.write("------------------------------------------------------\n")

pruned_flops = 0
pruned_ratio = 1. * pruning_ratio / pruning_step
pruned_layers_num = len(pruned_layers)
for i in range(pruned_layers_num):
    l_idx = pruned_layers[i]
    pruned_flops += flops_queue[l_idx]
pruned_flops *= pruned_ratio
pruned_flops = int(pruned_flops)

pruned_param = 0
for i in range(pruned_layers_num):
    l_idx = pruned_layers[i]
    pruned_param += l_param_queue[l_idx]
pruned_param *= pruned_ratio
pruned_param = int(pruned_param)



print("MAC: {}".format(flops))
print("MAC pruned: {}".format(pruned_flops))
print("MAC after prune: {}".format(flops-pruned_flops))
mac_decrease_ratio = 1. * pruned_flops / flops
print("MAC decrease: {:.6f}%".format(100 * mac_decrease_ratio))
print("------------------------------------------------------")
fresult.write("MAC: {}\n".format(flops))
fresult.write("MAC pruned: {}\n".format(pruned_flops))
fresult.write("MAC after prune: {}\n".format(flops-pruned_flops))
fresult.write("MAC decrease: {:.6f}%\n".format(100 * mac_decrease_ratio))
fresult.write("------------------------------------------------------\n")

print("PARAM: {}".format(param))
print("PARAM pruned: {}".format(pruned_param))
print("PARAM after prune: {}".format(param-pruned_param))
param_decrease_ratio = 1. * pruned_param / param
print("PARAM decrease: {:.6f}%".format(100 * param_decrease_ratio))
print("------------------------------------------------------")
fresult.write("PARAM: {}\n".format(param))
fresult.write("PARAM pruned: {}\n".format(pruned_param))
fresult.write("PARAM after prune: {}\n".format(param-pruned_param))
fresult.write("PARAM decrease: {:.6f}%\n".format(100 * param_decrease_ratio))
fresult.write("------------------------------------------------------\n")

fresult.close()




