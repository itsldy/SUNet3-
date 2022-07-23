# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:49:55 2022

@author: 17879
"""


from tensorflow.keras import  Model
from keras_preprocessing import image
import numpy as np
import time
from model import build_model
from unit import image_process
import random
from tensorflow.keras import  backend as K
from tensorflow.keras.models import load_model
import os

model = build_model([256,256,3]) #cdd[256,256,3] ,lv[512,512,3]
stra = './data/test/'
model.load_weights('./savemodel/weights_100--0.0097.h5')

def generator_data(batch_size,*args):#batch_size每个批次图片个数,**kwargs路径字典myarg2=path_train+'testA'
    p_a=image.list_pictures(args[0])#两个输入图片，和对应标签的路径列表为p_a,p_b,p_l
    p_b=image.list_pictures(args[1])
    #group_path = list(zip(p_a,p_b,p_l)) #组合乱序
    #random.shuffle(group_path)
    num = len(p_a)//batch_size#计算按照当前批次大小，需要的循环个数
    im_preprocess=image_process(256)
    i = 0
    while(True):
        filea = p_a[batch_size*i:batch_size*(i+1)]#读取一个批次的图片路径
        fileb = p_b[batch_size*i:batch_size*(i+1)]
        data_a=im_preprocess.image_read(filea)
        data_b=im_preprocess.image_read(fileb)
        if i == num:
            i = 0
            break
        i+=1
        yield [data_a,data_b]
        
pa = image.list_pictures('./data/test/label')
name = []
for p in pa:
    full_filename = os.path.basename(p)
    file_name, extension_name = os.path.splitext(full_filename)
    name.append(file_name)
j=0
batch = 2 
for f in generator_data(batch,stra+'A',stra+'B',stra+'label'):
    temp = model.predict(f)
    for i in range(batch):
        image.save_img('.//pre_image/'+name[i+j]+'.jpg',temp[-1][i])
    j+=batch
