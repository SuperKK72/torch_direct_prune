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



#sys.args
# prototxt_path = "./models/vgg16/deploy.prototxt"
# caffemodel_path = "./models/vgg16/deploy.caffemodel"

# prototxt_path = "./models/resnet_v1_50/deploy.prototxt"
# caffemodel_path = "./models/resnet_v1_50/deploy.caffemodel"

prototxt_path = "./models/mobilenet_v1/deploy.prototxt"
caffemodel_path = "./models/mobilenet_v1/deploy.caffemodel"

# prototxt_path = "./models/mobilenet_v2/deploy.prototxt"
# caffemodel_path = "./models/mobilenet_v2/deploy.caffemodel"

# prototxt_path = "./models/googlenet/deploy.prototxt"
# caffemodel_path = "./models/googlenet/deploy.caffemodel"

# prototxt_path = "./models/alexnet/deploy.prototxt"
# caffemodel_path = "./models/alexnet/deploy.caffemodel"

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
    images.append(os.path.join(valid_dataset_path, line_list[0]))
dataset.close()
# print(len(labels))
# val_dataset_file_path = "./valid.txt"
# for root, dirs, files in os.walk(valid_dataset_path):
#     # print(files)
#     cnt = len(files)
#     # print(cnt)
#     for i in range(cnt-1):
#         images.append(os.path.join(valid_dataset_path, files[i]))



image_num = len(images)
image_num = 100
caffe.set_mode_gpu()
net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)

top1_cnt = 0
top5_cnt = 0
log_path = "./log/log.txt"
#image_num = 1000
log_file = open(log_path, 'w')
for i in range(image_num):
    if i % 50 == 0:
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
        log_file.write('label: %d confidence: %.2f --- %s\n' % (label, prob[label], classes[label]))
    #print("---------------------------------")
    log_file.write("---------------------------------\n")
print("top1_cnt:{}\ttop5_cnt:{}\tall:{}\n".format(top1_cnt,top5_cnt,image_num))
top1_acc = 1.0 * top1_cnt / image_num
top5_acc = 1.0 * top5_cnt / image_num
print("top1 accuracy: {}    top5 accuracy: {}".format(top1_acc, top5_acc))
log_file.write("top1 accuracy: {}    top5 accuracy: {}\n".format(top1_acc, top5_acc))
log_file.close()





