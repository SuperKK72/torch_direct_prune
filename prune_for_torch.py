import torch
import torchvision.models as models
# from torchsummary import summary
import numpy as np
import os
from tool import torch_infer
os.environ["CUDA_VISION_DEVICES"] = "0,1"
'''
-----------------------------------------------------------------
version: torch-1.0.1.post2 + torchvision-0.2.0
this torchvision version include the following models:
resnet101, resnet18, 
alexnet,
densenet121, densenet161, densenet169, densenet201,
inception_v3, 
resnet34, resnet50, resnet152, 
squeezenet1_0, squeezenet1_1,
VGG11, VGG11_bn, VGG13, VGG13_bn, VGG16, VGG16_bn, VGG19, VGG19_bn,
-----------------------------------------------------------------
version: torch-1.18.1 + torchvision-0.9.1
this torchvision version include the following models:
inception_v3, googlenet, 
mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small,
vgg16, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16_bn, vgg19, vgg19_bn
resnet101, resnet50, resnet18, resnet34, resnet152, resnext50_32x4d, resnext101_32x8d,
squeezenet1_1, squeezenet1_0,
alexnet,
densenet121, densenet161, densenet169, densenet201,
mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3,
shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0,
wide_resnet50_2, wise_resnet101_2
-----------------------------------------------------------------
'''

def init_net():
    # net = models.resnet18(pretrained=True).cuda()
    # net = models.alexnet(pretrained=True).cuda()
    # net = models.squeezenet1_1(pretrained=True).cuda()
    # net = models.vgg16(pretrained=True).cuda()
    # net = models.resnet101(pretrained=True).cuda()
    net = models.mobilenet_v2(pretrained=True).cuda()
    return net
net = init_net()
net.eval()
# exit()

'''
param:
net_name: network name
valid_dataset_path: the folder of valid images
synset_path: label file
pruning_step: prune group size
pruning_ratio: prune num per group
'''
net_name = "mobilenet_v2"
valid_dataset_path = "./valid"
synset_path = "./synset.txt"
label_path = "./valid.txt"
pruning_step = 64
pruning_ratio = 16
result_file = "{}_{}_{}".format(net_name,pruning_ratio,pruning_step)
result_path = "./{}.txt".format(result_file)

fresult = open(result_path, 'w')
fresult.write("net name: {}\n".format(net_name))
fresult.write("prune mode: {}/{}\n".format(pruning_ratio,pruning_step))
fresult.write("------------------------------------------------------\n")

'''get images and labels'''
classes = np.loadtxt(synset_path, str, delimiter='\t')
images = []
labels = []
dataset = open(label_path,'r')
for line in dataset:
    line_list = line.strip().split(' ')
    labels.append(line_list[-1])
    images.append(os.path.join(valid_dataset_path, line_list[0]))
dataset.close()

'''Searching all conv layers can be pruned...'''
wts = net.state_dict()
print("Searching all conv layers...")
prune_layers_pre = []
flops = 0
flops_queue = []
param = 0
l_param_queue = []
for wt_name, wt_data in wts.items():
    print("{:40s} {}".format(wt_name, wt_data.shape))
    wt_shape = wts[wt_name].shape
    # ft_shape = 0
    if len(wt_data.shape)==4 and wt_data.shape[1] >= pruning_step:
        prune_layers_pre.append(wt_name)
        # print(type(wt_data))
        c_in = wt_shape[1]
        k_h = wt_shape[2]
        k_w = wt_shape[3]
        # c_out = ft_shape[1]
        # h_out = ft_shape[2]
        # w_out = ft_shape[3]
        # l_flops = c_out * h_out * w_out * c_in * k_h * k_w
        # flops_queue.append(l_flops)
        # flops += l_flops
        l_param = wt_shape[0] * wt_shape[1] * wt_shape[2] * wt_shape[3]
        l_param_queue.append(l_param)
        param += l_param
    elif len(wt_data.shape)==2 and wt_data.shape[1] >= pruning_step:
        prune_layers_pre.append(wt_name)
        # l_flops = wt_shape[0] * wt_shape[1]
        # flops_queue.append(l_flops)
        # flops += l_flops
        l_param = wt_shape[0] * wt_shape[1]
        l_param_queue.append(l_param)
        param += l_param
print("------------------------------------------------------")
print("all {} conv layers...".format(len(prune_layers_pre)))
print(prune_layers_pre)
print("------------------------------------------------------")
fresult.write("all {} conv params...\n".format(len(prune_layers_pre)))
for i in range(len(prune_layers_pre)):
    fresult.write(str(prune_layers_pre[i])+'\t')
    if i % 3 == 0:
        fresult.write('\n')
fresult.write('\n')
fresult.write("------------------------------------------------------\n")

'''get original top1 accuracy'''
image_num = 1000
top1_cnt_ori = 0
top1_acc_ori = 0.
print("get top1_acc_ori...")
for i in range(image_num):
    if i % 100 == 0:
        print("eval image {}/{}......".format(i, image_num))
    output = torch_infer(net, images[i])
    idx = np.argmax(output)
    if int(labels[i]) == idx:
        top1_cnt_ori = top1_cnt_ori + 1
top1_acc_ori = 1.0 * top1_cnt_ori / image_num
print("top1_cnt: {:3d}\ttop1_acc: {:.6f}".format(top1_cnt_ori, top1_acc_ori))
print("------------------------------------------------------")
fresult.write("top1_cnt: {:3d}\ttop1_acc: {:.6f}\n".format(top1_cnt_ori, top1_acc_ori))
fresult.write("------------------------------------------------------\n")

'''get top1_acc_queue'''
layer_num = len(prune_layers_pre)
top1_acc_queue = []
for l_idx in range(layer_num):
    # net = models.resnet18(pretrained=True).cuda()
    net = init_net()
    net.eval()
    wts = net.state_dict()
    print("pruning layer {} {}...".format(l_idx, prune_layers_pre[l_idx]))
    curr_wt = wts[prune_layers_pre[l_idx]].cpu().detach().numpy()
    # print(curr_wt[0,:,0,0])
    print(curr_wt.shape)
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
        wts[prune_layers_pre[l_idx]] = torch.Tensor(curr_wt)
        net.load_state_dict(wts)
        # net.state_dict().update(wts)
        # print(curr_wt[0, :, 0, 0])
        # print(net.state_dict()[prune_layers_pre[l_idx]][0, :, 0, 0])
        # exit()
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
        wts[prune_layers_pre[l_idx]] = torch.Tensor(curr_wt)
        net.load_state_dict(wts)
    top1_cnt = 0
    top1_acc = 0
    for i in range(image_num):
        if i%100==0:
            print("eval image {}/{}......".format(i, image_num))
        output = torch_infer(net, images[i])
        idx = np.argmax(output)
        if int(labels[i]) == idx:
            top1_cnt = top1_cnt + 1
    top1_acc = 1.0 * top1_cnt / image_num
    print("top1_cnt: {:3d}\ttop1_acc: {:.6f}".format(top1_cnt, top1_acc))
    top1_acc_queue.append(top1_acc)
    print("------------------------------------------------------")
print("prune top1_acc queue...")
print("all {} layers...".format(len(top1_acc_queue)))
print(top1_acc_queue)
print("------------------------------------------------------")
fresult.write("prune top1_acc queue...\n")
fresult.write("all {} layers...\n".format(len(top1_acc_queue)))
fresult.write(str(top1_acc_queue)+'\n')
fresult.write("------------------------------------------------------\n")

'''get prune queue'''
top1_baseline = top1_acc_ori
print("original top1_acc: {}".format(top1_baseline))
print("------------------------------------------------------")
prune_queue = []
prune_queue_pre = np.argsort(top1_acc_queue)
prune_queue_pre = prune_queue_pre[::-1]
print("prune queue pre...")
print("all {} layers...".format(len(prune_queue_pre)))
print(prune_queue_pre)
print("------------------------------------------------------")
#find layers that l_top1_acc_decrease <= 0.5
for i in range(layer_num):
    err = top1_acc_queue[prune_queue_pre[i]] - top1_baseline
    if err >= -0.005:
        prune_queue.append(prune_queue_pre[i])
    # if top1_acc_queue[prune_queue_pre[i]] >= 0.7:
    #     prune_queue.append(prune_queue_pre[i])
print("prune queue...")
print("all {} layers...".format(len(prune_queue)))
print(prune_queue)
print("------------------------------------------------------")
fresult.write("prune queue...\n")
fresult.write("all {} layers...\n".format(len(prune_queue)))
fresult.write(str(prune_queue)+'\n')
fresult.write("------------------------------------------------------\n")

'''prune the network'''
#find the prune_layer_queue that n_top1_acc_decrease >= 0.1 or 0.2
prune_queue_len = len(prune_queue)
# net = models.resnet18(pretrained=True).cuda()
net = init_net()
net.eval()
wts = net.state_dict()
pruned_layers = []
pruned_top1_acc = []
pruned_params = []
for p_idx in range(prune_queue_len):
    l_idx = prune_queue[p_idx]
    print("pruning layer {} {}...".format(l_idx, prune_layers_pre[l_idx]))
    '''amazing....jing ran zhi qian mei baocuo...'''
    curr_wt = wts[prune_layers_pre[l_idx]].cpu().detach().numpy()
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
        wts[prune_layers_pre[l_idx]] = torch.Tensor(curr_wt)
        net.load_state_dict(wts)
        # net.state_dict().update(wts)
        # print(curr_wt[0, :, 0, 0])
        # print(net.state_dict()[prune_layers_pre[l_idx]][0, :, 0, 0])
        # exit()
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
    top1_acc = 0
    for i in range(image_num):
        # if i%100==0:
        #     print("eval image {}/{}......".format(i, image_num))
        output = torch_infer(net, images[i])
        idx = np.argmax(output)
        if int(labels[i]) == idx:
            top1_cnt = top1_cnt + 1
    top1_acc = 1.0 * top1_cnt / image_num
    if (top1_acc - top1_baseline) < -0.01:
        pruned_top1_acc.append(top1_acc)
        break
    pruned_layers.append(l_idx)
    pruned_top1_acc.append(top1_acc)
    pruned_params.append(prune_layers_pre[l_idx])
    print("top1_cnt: {:3d}\ttop1_acc: {:.6f}".format(top1_cnt, top1_acc))
    print("------------------------------------------------------")
print("pruned top1_acc...")
print("all {} layers...".format(len(pruned_top1_acc)))
print(pruned_top1_acc)
print("------------------------------------------------------")
print("pruned layers...")
print("all {} layers...".format(len(pruned_layers)))
print(pruned_layers)
print("------------------------------------------------------")
fresult.write("pruned top1_acc...\n")
fresult.write("all {} layers...\n".format(len(pruned_top1_acc)))
fresult.write(str(pruned_top1_acc)+'\n')
fresult.write("------------------------------------------------------\n")
fresult.write("pruned param ids...\n")
fresult.write("all {} params...\n".format(len(pruned_layers)))
fresult.write(str(pruned_layers)+'\n')
fresult.write("------------------------------------------------------\n")
fresult.write("pruned param names...\n")
fresult.write("all {} params...\n".format(len(pruned_params)))
for i in range(len(pruned_params)):
    fresult.write(str(pruned_params[i])[:-7]+', ')
fresult.write('\n')
fresult.write("------------------------------------------------------\n")

'''mac'''
# pruned_flops = 0
pruned_ratio = 1. * pruning_ratio / pruning_step
pruned_layers_num = len(pruned_layers)
# for i in range(pruned_layers_num):
#     l_idx = pruned_layers[i]
#     pruned_flops += flops_queue[l_idx]
# pruned_flops *= pruned_ratio
# pruned_flops = int(pruned_flops)
# print("MAC: {}".format(flops))
# print("MAC pruned: {}".format(pruned_flops))
# print("MAC after prune: {}".format(flops-pruned_flops))
# mac_decrease_ratio = 1. * pruned_flops / flops
# print("MAC decrease: {:.6f}%".format(100 * mac_decrease_ratio))
# print("------------------------------------------------------")

'''param'''
pruned_param = 0
for i in range(pruned_layers_num):
    l_idx = pruned_layers[i]
    pruned_param += l_param_queue[l_idx]
pruned_param *= pruned_ratio
pruned_param = int(pruned_param)
param_decrease_ratio = 1. * pruned_param / param
print("PARAM: {}".format(param))
print("PARAM pruned: {}".format(pruned_param))
print("PARAM after prune: {}".format(param-pruned_param))
print("PARAM decrease: {:.6f}%".format(100 * param_decrease_ratio))
print("------------------------------------------------------")
fresult.write("PARAM: {}\n".format(param))
fresult.write("PARAM pruned: {}\n".format(pruned_param))
fresult.write("PARAM after prune: {}\n".format(param-pruned_param))
fresult.write("PARAM decrease: {:.6f}%\n".format(100 * param_decrease_ratio))
fresult.write("------------------------------------------------------\n")



fresult.close()