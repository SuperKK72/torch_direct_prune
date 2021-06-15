import argparse
# import caffe
import numpy as np
import cv2
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description='evaluate pretrained mobilenet models')
    parser.add_argument('--prototxt', dest='prototxt',
                        help="Path of prototxt file.", type=str)
    parser.add_argument('--caffemodel', dest='caffemodel',
                        help='Path of caffemodel file.', type=str)
    parser.add_argument('--valid_dataset', dest='valid_dataset',
                        help='Path of valid dataset.', type=str)
    parser.add_argument('--synset', dest='synset',
                        help="Path of synset file.", type=str)
    parser.add_argument('--label', dest='label',
                        help="Path of label file.", type=str)
    # parser.add_argument('--log')
    # parser.add_argument('--output')

    args = parser.parse_args()
    return args, parser

def eval(net, image_path):
    input_h, input_w = 224, 224
    # mean_value = np.array([103.94, 116.78, 123.68], dtype=np.float32)
    # mean_value = np.array([128, 128, 128], dtype=np.float32)

    image = caffe.io.load_image(image_path)
    image = caffe.io.resize_image(image, [input_h, input_w])

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # row to col
    transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR
    transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
    # transformer.set_mean('data', mean_value)
    # transformer.set_input_scale('data', 0.0078125)
    # transformer.set_input_scale('data', 0.017)


    net.blobs['data'].reshape(1, 3, input_h, input_w)
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    out = net.forward()
    # prob = out['prob']
    prob = out['prob']
    prob = np.squeeze(prob)
    idx = np.argsort(-prob)
    return prob, idx

def torch_infer(net, img_path):
    image_h = 224
    image_w = 224
    mean_R = 0.485
    mean_G = 0.456
    mean_B = 0.406
    std_R = 0.229
    std_G = 0.224
    std_B = 0.225
    image = cv2.imread(img_path, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_h, image_w))
    image = np.array(image).astype(np.float32)
    image[:, :, 0] = (image[:, :, 0]/255 - mean_R) / std_R
    image[:, :, 1] = (image[:, :, 1]/255 - mean_G) / std_G
    image[:, :, 2] = (image[:, :, 2]/255 - mean_B) / std_B
    image = image.transpose((2,0,1))
    image = image.reshape(1,3,image_h,image_w)
    image = torch.Tensor(image).cuda()
    output = net.forward(image)
    output = output.cpu().detach().numpy().flatten()
    return output