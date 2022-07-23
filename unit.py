# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:46:35 2021

@author: DELL
"""

import tensorflow as tf 
from tensorflow.keras import  layers
from tensorflow.keras.layers import  Activation, Conv2D
from tensorflow.keras.layers import GlobalMaxPool2D, GlobalAveragePooling2D, Dense, Reshape, Concatenate
from keras_preprocessing import image
import numpy as np
import random

def CBAM(x,nb_filter, stage):
    # Channel Attention
    avgpool = GlobalAveragePooling2D(name=stage+'_channel_avgpool')(x)
    maxpool = GlobalMaxPool2D(name=stage+'_channel_maxpool')(x)
    # Shared MLP
    Dense_layer1 = Dense(nb_filter//8, activation='relu', name=stage+'_channel_fc1')
    Dense_layer2 = Dense(nb_filter, activation='relu', name=stage+'_channel_fc2')
    avg_out = Dense_layer2(Dense_layer1(avgpool))
    max_out = Dense_layer2(Dense_layer1(maxpool))

    channel = layers.add([avg_out, max_out])
    channel = Activation('sigmoid', name=stage+'_channel_sigmoid')(channel)
    channel = Reshape((1,1,nb_filter), name=stage+'_channel_reshape')(channel)
    channel_out = layers.Multiply()([x, channel])

    # Spatial Attention
    #avgpool = tf.reduce_mean(channel_out, axis=3, keepdims=True, name=stage+'_spatial_avgpool')
    avgpool =  layers.Lambda(tf.reduce_mean,arguments={'axis':3,'keepdims':True,'name':stage+'_spatial_avgpool'})(channel_out)
    #maxpool = tf.reduce_max(channel_out, axis=3, keepdims=True, name=stage+'_spatial_maxpool')
    maxpool  =  layers.Lambda(tf.reduce_max,arguments={'axis':3,'keepdims':True,'name':stage+'_spatial_avgpool'})(channel_out)
    spatial = Concatenate(axis=3)([avgpool, maxpool])

    spatial = Conv2D(1, (7,7), strides=1, padding='same',name=stage+'_spatial_conv2d')(spatial)
    spatial_out = Activation('sigmoid', name=stage+'_spatial_sigmoid')(spatial)

    CBAM_out = layers.Multiply()([channel_out, spatial_out])
    return CBAM_out


class image_process():
    def __init__(self,image_size=256):
        self.image_size = image_size
    
    def __generate_mask(self,mode="clip"):#生成256*256的掩膜矩阵，mode其他值时掩膜值为1
        if mode=="clip":
            line = np.random.randint(0,2,size=(32,32))
            repeat_weight = np.repeat(line,8,axis=1)
            repeat_height = np.repeat(repeat_weight,8,axis=0)
            gen_mask = repeat_height.reshape((256,256,1))
        else:
            gen_mask = np.ones((256,256,1))
        return gen_mask
    
    def image_read(self,file_path,not_label=True):#根据路径列表读取图片
        bs = []#为将每张图片合并为一个数组
        if not_label:
            for i in range(len(file_path)):
                data = image.load_img(file_path[i],color_mode='rgb')# 返回 ndarray
                data = data.resize((self.image_size,self.image_size))
                img_array = image.img_to_array(data)
                bs.append(img_array[np.newaxis])
            x = np.concatenate(bs,axis=0)#按照0轴连接
        else:
            for i in range(len(file_path)):
                data = image.load_img(file_path[i],color_mode='grayscale')# 返回 ndarray
                data = data.convert('1')
                data = data.resize((self.image_size,self.image_size))
                img_array = image.img_to_array(data)
                bs.append(img_array[np.newaxis])
            x = np.concatenate(bs,axis=0)#按照0轴连接
        return x
    
    def mask_image(self,file_path,not_label=True,mode="clip"):#利用掩膜层掩膜图像
        bs = []
        maskchannel = self.__generate_mask(mode)
        image_array = self.image_read(file_path,not_label)
        b,w,h,channel = image_array.shape
        mask_array =np.repeat(maskchannel,channel,axis=-1)
        for i in range(len(image_array)):
            mask_out = image_array[i]*mask_array
            bs.append(mask_out[np.newaxis])
        return np.concatenate(bs,axis=0)
            
    def add_mask(self,file_path,not_label=True,mode="clip"):#给图像增加掩膜层
        bs = []
        maskchannel = self.__generate_mask(mode)
        image_array = self.image_read(file_path,not_label)
        for i in range(len(image_array)):
            mask_out = np.concatenate([image_array[i],maskchannel],axis=-1)
            bs.append(mask_out[np.newaxis])
        return np.concatenate(bs,axis=0)
    
def generator_easy(batch_size,*args):#batch_size每个批次图片个数,**kwargs路径字典myarg2=path_train+'testA'
    p_a=image.list_pictures(args[0])#两个输入图片，和对应标签的路径列表为p_a,p_b,p_l
    p_b=image.list_pictures(args[1])
    p_l = image.list_pictures(args[2])
    group_path = list(zip(p_a,p_b,p_l)) #组合乱序
    random.shuffle(group_path)
    num = len(p_a)//batch_size#计算按照当前批次大小，需要的循环个数
    p_a,p_b,p_l = zip(*group_path) #拆分
    im_preprocess=image_process(image_size=256)#cdd for 256. lv for 512
    i = 0
    while(True):
        filea = p_a[batch_size*i:batch_size*(i+1)]#读取一个批次的图片路径
        fileb = p_b[batch_size*i:batch_size*(i+1)]
        filel = p_l[batch_size*i:batch_size*(i+1)]
        data_a=im_preprocess.image_read(filea)
        data_b=im_preprocess.image_read(fileb)
        label_y = im_preprocess.image_read(filel,not_label=False)
        i+=1
        if i == num:
            i = 0
            random.shuffle(group_path)
            p_a,p_b,p_l = zip(*group_path) #重新拆分
        yield [data_a,data_b],[label_y,label_y,label_y,label_y,label_y]
        


        
