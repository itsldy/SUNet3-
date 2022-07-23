# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:39:47 2021

@author: 17879
"""

from tensorflow.keras import backend as K
import tensorflow as tf

# loss

def IOU(y_true, y_pred, smooth=1):

    IoU = 0.0
    #compute the IoU of the foreground
    Iand1 =  K.sum(y_true*y_pred,axis=(1,2,-1))
    Ior1 =  K.sum(y_true,axis=(1,2,-1)) +  K.sum(y_pred,axis=(1,2,-1))-Iand1
    IoU1 = (Iand1+smooth)/(Ior1+smooth)
    #IoU loss is (1-IoU1)
    IoU += (1-IoU1) * smooth

    return K.mean(IoU )

def focal_loss(y_true, y_pred,gamma=2.0,alpha= 0.25):

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)
    focal = -alpha * K.pow(1. -pt_1, gamma) * K.log(pt_1)-(1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 )
    #focal = -alpha * K.pow(1. -pt_1, gamma) * K.log(pt_1)-alpha * K.pow( pt_0, gamma) * K.log(1. - pt_0 )
    #总的loss
    loss = K.mean(focal)
    return loss



def tf_ssim_loss(y_true,y_pred):
    
    total_loss=1-tf.image.ssim_multiscale(y_true,y_pred,max_val=255)
    
    return K.mean(total_loss)  

def dice_coef(y_true, y_pred, smooth=1, weight=0.5):
    
    """
    加权后的dice coefficient
    """
    y_true = y_true[:, :, :, -1]  # y_true[:, :, :, :-1]=y_true[:, :, :, -1] if dim(3)=1 等效于[8,256,256,1]==>[8,256,256]
    y_pred = y_pred[:, :, :, -1]
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + weight * K.sum(y_pred)
    # K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    return ((2. * intersection + smooth) / (union + smooth))  # not working better using mean


def dice_coef_loss(y_true, y_pred):
    
    """
    目标函数
    """
    return 1 - dice_coef(y_true, y_pred)

def weighted_bce_dice_loss(y_true,y_pred):
    class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])

    class_weights = [0.1, 0.9]#note that the weights can be computed automatically using the training smaples
    weighted_bce = K.sum(class_loglosses * K.constant(class_weights))

    # return K.weighted_binary_crossentropy(y_true, y_pred,pos_weight) + 0.35 * (self.dice_coef_loss(y_true, y_pred)) #not work
    return weighted_bce + 0.5 * (dice_coef_loss(y_true, y_pred))


def bce_dice_loss(y_true,y_pred):
    class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])

    #class_weights = [0.1, 0.9]#note that the weights can be computed automatically using the training smaples
    weighted_bce = K.sum(class_loglosses)

    # return K.weighted_binary_crossentropy(y_true, y_pred,pos_weight) + 0.35 * (self.dice_coef_loss(y_true, y_pred)) #not work
    return K.mean(weighted_bce + (dice_coef_loss(y_true, y_pred)))

def focal_dice_loss(y_true,y_pred):
    y_t = y_true[:, :, :, -1]
    weight_1 = tf.where(tf.equal(y_t, 0), 0.0,0.8)
    weight_0 = tf.where(tf.equal(y_t, 1), 0.0,0.2)
    weight_all = weight_1+weight_0
    loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2)
    focal = loss(y_true, y_pred, sample_weight= weight_all)
    # focal = tf.keras.losses.BinaryFocalCrossentropy(gamma=3, from_logits=True)
    # loss = focal(y_true, y_pred, sample_weight=[0.8, 0.2])
    return focal + 0.5 * (dice_coef_loss(y_true, y_pred))