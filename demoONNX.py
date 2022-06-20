# coding:utf-8
from __future__ import print_function
import imp
import torch
from torchvision import transforms
from utils import *
import cfgs.cfgs_eval as cfgs
from collections import OrderedDict
import os
import onnx
import onnxruntime as ort
from PIL import Image

def Train_or_Eval(model, state='Train'):
    if state == 'Train':
        model.train()
    else:
        model.eval()


img_width = 256
img_height = 64
image_path = './demo/'
output_txt = './demo/predictions.txt'
image_list = os.listdir(image_path)
transf = transforms.ToTensor()
test_acc_counter = Attention_AR_counter('\ntest accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                            cfgs.dataset_cfgs['case_sensitive'])

model_ONNX = onnx.load("VisionLAN.onnx")
onnx.checker.check_model(model_ONNX)

ort_sess = ort.InferenceSession('VisionLAN.onnx')

for img_name in image_list:
        img = Image.open(image_path + img_name).convert('RGB')
        img = img.resize((img_width, img_height))
        img = transf(img)
        img = torch.unsqueeze(img,dim = 0)
        target = ''
        #output, out_length = model(img, target, '', False)
        output = ort_sess.run(None, {'input': img.numpy()})
        #pre_string = test_acc_counter.convert(output, out_length='')
        #print('pre_string:',pre_string[0])
        #print('output:',output[0])