# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 21:42:16 2022

@author: 17879
"""

from keras_preprocessing import image
import numpy as np
import tensorflow as tf
from model import build_model
from unit import generator_easy
from tensorflow.keras.optimizers import Adam
from allloss_keras import weighted_bce_dice_loss

ss = build_model([256,256,3])

checkpoint_filepath1 = './save_model/weights_{epoch:03d}-{val_loss:.4f}.h5'
model_checkpoint_callback1 = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_filepath1, monitor='val_loss', verbose=1, save_best_only=False,
    save_weights_only=True, mode='auto', save_freq='epoch',
    options=None)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001,patience=30)

path_train = './/train/'#change dataset root
path_val = './/val/'

# cdd数据集编码器的训练
ss.fit_generator(generator_easy(8,path_train+'A',path_train+'B',path_train+'label'), 
                                steps_per_epoch=1250, epochs=300,verbose=1,callbacks=[model_checkpoint_callback1],
                                validation_data=generator_easy(8,path_val+'A',path_val+'B',path_val+'label'), 
                                validation_steps=375)
  
