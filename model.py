# -*- coding: utf-8 -*-
"""
Created on Sun May  8 15:58:05 2022

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May  8 15:40:17 2022
model for siam
@author: 17879
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Conv2D,BatchNormalization,Add,DepthwiseConv2D,Activation,UpSampling2D
from tensorflow.keras.layers import Conv2DTranspose,concatenate,MaxPooling2D,Subtract, Reshape, dot,add,Lambda
from tensorflow.keras import backend as K
from allloss_keras import weighted_bce_dice_loss,focal_dice_loss
from tensorflow.keras.models import load_model
from tensorflow import keras
from functools import reduce

class resblock(layers.Layer):
    def __init__(self,channel=64,kernel=(3,3),**kwargs):
        super(resblock, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.x0 = Conv2D(self.channel, self.kernel , activation='relu', name='conv0',kernel_initializer='he_normal', padding='same')
        self.x1 = BatchNormalization(name='bn0')
        self.x2 = Conv2D(self.channel, self.kernel , activation='relu', name='conv1',kernel_initializer='he_normal', padding='same')
        self.x3 = BatchNormalization(name='bn1')
        self.x4 = Add(name='resi')
    def call(self, x):
        feature = self.x4([self.x3(self.x2(self.x1(self.x0(x)))),self.x0(x)])
        return feature
    def get_config(self):
        config = super(resblock, self).get_config()
        config.update({'channel': self.channel,'kernel': self.kernel})
        return config

class none_local_block(layers.Layer):
    def __init__(self,channel=64,kernel=(1,1),**kwargs):
        super(none_local_block,self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.fk = Conv2D(int(self.channel/2),self.kernel, padding='same', use_bias=False, kernel_initializer='he_normal',name='f_ka')
        self.fq = Conv2D(int(self.channel/2),self.kernel, padding='same', use_bias=False, kernel_initializer='he_normal',name='f_qa') 
        self.fv = Conv2D(int(self.channel/2),self.kernel, padding='same', use_bias=False, kernel_initializer='he_normal',name='f_vb')
        self.out = Conv2D(self.channel,self.kernel, padding='same', use_bias=False, kernel_initializer='he_normal',name='f_out')
    def call(self,x):
        batchsize, dim1, dim2, channels = K.int_shape(x[0])
        f_k = self.fk(x[0])
        reshape_f_k = Reshape((dim1*dim2, int(self.channel/2)),name='reshaep_k_fa')(f_k)
        f_q = self.fq(x[0]) 
        reshape_f_q = Reshape((dim1*dim2, int(self.channel/2)),name='reshaep_q_fa')(f_q)
        f_dot = dot([reshape_f_k, reshape_f_q], axes=2,name='dot_fk_fq')
        f_dot = layers.Softmax()(f_dot)
        f_v1 = self.fv(x[1])#imageb
        reshape_f_v1 = Reshape((dim1*dim2, int(self.channel/2)),name='reshaep_f_v1')(f_v1)
        generate_image1 = dot([f_dot,reshape_f_v1],axes=[2,1],name='generate_image1')
        reshape_image1 = Reshape((dim1,dim2, int(self.channel/2)),name='reshape_image1')(generate_image1)
        image1 = self.out(reshape_image1)
        return image1
    def get_config(self):
        config = super(none_local_block, self).get_config()
        config.update({
            'channel': self.channel,
            'kernel': self.kernel,
            })
        return config


class struct_layer(layers.Layer):
    def __init__(self,channel=64,kernel=(16,16),out_channel=3,**kwargs):
        super(struct_layer,self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.out_channel = out_channel
        self.pgm = Conv2D(self.channel,self.kernel,strides=self.kernel, use_bias=False, kernel_initializer='he_normal',name='pgm')
        self.tpgm = Conv2DTranspose(self.out_channel,self.kernel,strides=self.kernel, use_bias=False, kernel_initializer='he_normal',name='tpgm')
        self.nlb = none_local_block(self.channel)
    def call(self,x):
        p_im1 = self.pgm(x[0])
        p_im2 = self.pgm(x[1])
        ps1_g2 = self.nlb([p_im1,p_im2])
        sub2 = K.abs(Subtract(name='dif_b')([p_im2, ps1_g2]))
        marge_sub = self.tpgm(sub2)
        conc_sub = add([x[1],marge_sub])
        return conc_sub
    def get_config(self):
        config = super(struct_layer, self).get_config()
        config.update({
            'out_channel': self.out_channel,
            'channel': self.channel,
            'kernel': self.kernel,
            })
        return config
    
def build_encoder(features,num_channels=[32,64,128,256,512]):
    image1 ,image2 = features
    res0 = resblock(num_channels[1])#残差单元
    res1 = resblock(num_channels[2])
    res2 = resblock(num_channels[3])
    res3 = resblock(num_channels[4])
    res4 = resblock(num_channels[4])
    struct0 = struct_layer(64,(16,16),out_channel=64)#结构单元
    #第1层
    # feature_struct = struct0([image1,image2])
    feature0_0 = res0(image1)
    maxpool0_0 = MaxPooling2D((2,2))(feature0_0)
    feature0_1 = res0(image2)
    maxpool0_1 = MaxPooling2D((2,2))(feature0_1)
    features_0 = struct0([feature0_0,feature0_1])
    # features_0 = K.abs(Subtract()([feature0_0, feature0_1]))
    #第2层
    feature1_0 = res1(maxpool0_0)
    maxpool1_0 = MaxPooling2D((2,2))(feature1_0)
    feature1_1 = res1(maxpool0_1)
    maxpool1_1 = MaxPooling2D((2,2))(feature1_1)
    # features_1 = struct1([feature1_0,feature1_1])
    features_1 = K.abs(Subtract()([feature1_0, feature1_1]))
    #第3层
    feature2_0 = res2(maxpool1_0)
    maxpool2_0 = MaxPooling2D((2,2))(feature2_0)
    feature2_1 = res2(maxpool1_1)
    maxpool2_1 = MaxPooling2D((2,2))(feature2_1)
    # features_2 = struct2([feature2_0,feature2_1])
    features_2 = K.abs(Subtract()([feature2_0,feature2_1]))
    #第4层
    concate_feature3 = concatenate([maxpool2_0,maxpool2_1])
    feature3 = res3(concate_feature3)

    maxpool_feature3 = MaxPooling2D((2,2))(feature3)
    feature4 = res4(maxpool_feature3)
    return[features_0,features_1,features_2,feature3,feature4]


def build_decoder(features):
    feature0_2,feature1_2,feature2_2,feature3_2 = features

        # DeepSup
    # out_pic0 = Conv2D(1, (1, 1), name='out_pic0',
    #                   kernel_initializer='he_normal', padding='same')(feature_struct)
    # out0 = Activation('sigmoid',name='out0')(out_pic0)
    out_pic1 = Conv2D(1, (1, 1), name='out_pic1',
                      kernel_initializer='he_normal', padding='same')(feature0_2)
    out1 = Activation('sigmoid',name='out1')(out_pic1)
    
    out_pic2 = Conv2D(1, (1, 1), name='out_pic2',
                      kernel_initializer='he_normal', padding='same')(feature1_2)
    up_out_pic2 = UpSampling2D(size=(2, 2), interpolation='bilinear')(out_pic2)
    out2 = Activation('sigmoid',name='out2')(up_out_pic2)
    
    out_pic3 = Conv2D(1, (1, 1), name='out_pic3',
                      kernel_initializer='he_normal', padding='same')(feature2_2)
    up_out_pic3 = UpSampling2D(size=(4, 4), interpolation='bilinear')(out_pic3)
    out3 = Activation('sigmoid',name='out3')(up_out_pic3)
    
    out_pic4 = Conv2D(1, (1, 1), name='out_pic4',
                      kernel_initializer='he_normal', padding='same')(feature3_2)
    up_out_pic4 = UpSampling2D(size=(8, 8), interpolation='bilinear')(out_pic4)
    out4 = Activation('sigmoid',name='out4')(up_out_pic4)
    ## -------------msof-------------
    conv_fuse = concatenate([out1,out2,out3,out4], name='merge_fuse', axis=-1)
    out5 = Conv2D(1, (1, 1), activation='sigmoid', name='output_5',
                              kernel_initializer='he_normal', padding='same',)(conv_fuse)
    
    return [out1,out2,out3,out4,out5]

def marge_net(features,num_channels=[32,64,128,256,512]):
    # unet 加深一层
    features_0,features_1,features_2,feature_3,feature4 =features
    res0 = resblock(num_channels[1])#残差单元
    res1 = resblock(num_channels[2])
    res2 = resblock(num_channels[3])
    res3 = resblock(num_channels[4])
    features0_1 = Conv2D(128,(3,3),strides=(2,2),kernel_initializer='he_normal', padding='same')(features_0)
    features0_2 = Conv2D(256,(3,3),strides=(4,4),kernel_initializer='he_normal', padding='same')(features_0)
   
    features1_0 =  UpSampling2D(interpolation='bilinear')(features_1)
    features1_0 =  Conv2D(64,(3,3),kernel_initializer='he_normal', padding='same')(features1_0) 
    features1_2 = Conv2D(256,(3,3),strides=(2,2),kernel_initializer='he_normal', padding='same')(features_1)
    
    features2_0 =  UpSampling2D(size=(4, 4),interpolation='bilinear')(features_2)
    features2_0 =  Conv2D(64,(3,3),kernel_initializer='he_normal', padding='same')(features2_0)  
    features2_1 = UpSampling2D(interpolation='bilinear')(features_2)
    features2_1 =  Conv2D(128,(3,3),kernel_initializer='he_normal', padding='same')(features2_1)  
  
    features4_3 = UpSampling2D(size=(2,2),interpolation='bilinear')(feature4)
    feature_marge3 = add([features4_3,feature_3]) 
    feature3_2 = res3(feature_marge3)
    
    feature_marge3_2 = UpSampling2D(size=(2,2),interpolation='bilinear')(feature3_2)
    feature_marge3_2 = Conv2D(256,(3,3),kernel_initializer='he_normal', padding='same')(feature_marge3_2)  
    feature_marge2 = add([feature_marge3_2,features_2,features1_2,features0_2])
    feature2_2 = res2(feature_marge2)
    
    feature_marge2_1 = UpSampling2D(size=(2,2),interpolation='bilinear')(feature2_2)
    feature_marge2_1 = Conv2D(128,(3,3),kernel_initializer='he_normal', padding='same')(feature_marge2_1)  
    feature_marge1 = add([feature_marge2_1,features2_1,features_1,features0_1]) 
    feature1_2 = res1(feature_marge1) 
    
    feature_marge1_0 = UpSampling2D(size=(2,2),interpolation='bilinear')(feature1_2)
    feature_marge1_0 = Conv2D(64,(3,3),kernel_initializer='he_normal', padding='same')(feature_marge1_0)  
    feature_marge0 = add([feature_marge1_0,features2_0,features1_0,features_0]) 
    feature0_2 = res0(feature_marge0)
    
    return [feature0_2,feature1_2,feature2_2,feature3_2]

def build_model(input_tensor=[512,512,3]):# cdd[256,256,3] , lv[512,512,3]
    im1= Input(input_tensor)
    im2 = Input(input_tensor)
    feat = [im1,im2]
    x = build_encoder(feat)
    x = marge_net(x)
    x = build_decoder(x)
    model = Model(inputs=feat,outputs=x)
    model.compile(optimizer= Adam(1e-4), loss=[weighted_bce_dice_loss,weighted_bce_dice_loss,weighted_bce_dice_loss,
                                 weighted_bce_dice_loss,weighted_bce_dice_loss],
                   metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    model.summary()
    return model  

