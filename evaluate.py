# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:53:34 2021

@author: DELL
"""

def train_process(file_path):#将图像读取为数组
    data = image.load_img(file_path,color_mode='grayscale')# 返回 ndarray
    data = data.convert('1')
    #data = data.resize((512,512))
    img_array = image.img_to_array(data)
    return img_array

import numpy as np
from keras_preprocessing import image



pre_img_path = './/pre_CDD_16'
# pre_img_path = './/pre_image_LEVIR'
file_pre = image.list_pictures(pre_img_path)
true_img_path = 'D:/ldyuse/subset/test/label'
# true_img_path = 'D:/ldyuse/LEVIR_CD/test_512/label'
file_true = image.list_pictures(true_img_path)
# file_pre = [] 
# for i in range(len(file_true )):
#     strr = pre_img_path+np.str(i)+'.jpg'
#     file_pre.append(strr)
zipped_path = zip(file_pre,file_true)
sum_tp,sum_tn,sum_fp,sum_fn=[],[],[],[]
for (f1,f2) in zipped_path:         
    p_img=train_process(f1)
    t_img=train_process(f2)
    Add_train = p_img + t_img
    Sub_train = p_img - t_img
    TP_train = np.sum(np.where(Add_train==2,1,0))
    TN_train = np.sum(np.where(Add_train==0,1,0))
    FP_train = np.sum(np.where(Sub_train==1,1,0))
    FN_train = np.sum(np.where(Sub_train==-1,1,0))
    sum_tp.append(TP_train)
    sum_tn.append(TN_train)
    sum_fp.append(FP_train)
    sum_fn.append(FN_train)
tp = np.sum(np.array(sum_tp))
tn = np.sum(np.array(sum_tn))
fp = np.sum(np.array(sum_fp))
fn = np.sum(np.array(sum_fn))


precision = round(tp/(tp+fp),5)
recall = round(tp/(tp+fn),5)
Accuracy = round((tp+tn) / (tp+tn+fp+fn),5)
F1 = round(2*(precision*recall)/(precision+recall),5)
Iou = round(tp/(fn+fp+tp),5)
print('precision,recall,Accuracy,F1,Iou,is:{}'.format((precision,recall,Accuracy,F1,Iou)))


import matplotlib.pyplot as plt  
labels = ['precision','recall','F1','Accuracy','Iou']  
model_score = [precision,recall,F1,Accuracy,Iou] 
x = np.arange(len(labels))  # the label locations
width = 0.5  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x , model_score, width, label='LEVIR_CD',color=['deepskyblue'])#label = CDD OR LEVIR_CD
ax.set_ylabel('Scores')
ax.set_title('change detection results on LEVIR_CD datasates')# CDD OR LEVIR_CD
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for x1,y1 in enumerate(model_score):
    plt.text(x1, y1+0.01, y1,ha='center',fontsize=10)
