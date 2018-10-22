

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 13:48:52 2018

@author: danlinpeng
"""
import mne
import numpy as np
import torch
from scipy import signal, fftpack
import matplotlib.pyplot as plt

from mne.time_frequency.tfr import morlet
from mne.viz import plot_filter, plot_ideal_filter

import mne
import pickle

def get_epochs(filename):
    print(filename)
    #get raw data
    raw = mne.io.read_raw_cnt(filename,montage=None)
    raw.load_data()
    channel = raw.info['ch_names']
    raw.info['bads'] = ['M1'] 
    #design filter
    iir_params = dict(order=2, ftype='butter', output='sos')
    iir_params = mne.filter.construct_iir_filter(iir_params, [14,71], None, 1000, 'bandpass') 
    raw.filter(l_freq = 14.,h_freq = 71 ,method='iir',picks= mne.pick_channels(raw.info["ch_names"],raw.info["ch_names"][:-1]),iir_params=iir_params,phase='zero-double' )
# =============================================================================
#     raw.filter(l_freq = 14.,h_freq = 71 ,method='fir' )
# =============================================================================
# =============================================================================
#     mne.filter.filter_data(raw,1000,14,71,method='iir',iir_params=iir_params,phase='zero-double',)
# =============================================================================
    #extract epoch
    events = mne.find_events(raw)
    labels = events[:,2]
    tmin, tmax = 0.0, 0.549
    baseline = (None, 0.0)
    epochs = mne.Epochs(raw, events=events, tmin=tmin,
                    tmax=tmax, baseline=baseline)
    epochs.load_data()
    data = epochs.get_data() 


    #remove channel 32
    #remove channel 32
    data = np.delete(data, 32, axis = 1)
    data = np.delete(data, 41, axis = 1)
    #remove incorrect data
    temp = []
    temp2 = []
    
    for i in range(data.shape[0]):
        if data[i,64,0]<60:
            temp.append(data[i,:,:])
    data = np.array(temp)
    return data



def load_data(batch_size):
    event = [x for x in range(55)]
    for subject_num in [1]:
            #get data
            filename = "presentation/S"+str(subject_num)+"_Exp1-"+str(1)+".cnt"
            data = get_epochs(filename)
            for j in range(2,6):
                filename = "presentation/S"+str(subject_num)+"_Exp1-"+str(j)+".cnt"
                data = np.vstack((data,get_epochs(filename)))
            #pick event
            labels = [x for x in data [:,64,0]]
            index = np.isin(labels,event)
            index = np.where(index)
            data = data[index]*1000000
            labels = np.array(labels)
            labels = labels[index]
    data = data [:,:64,50:] 
    for k in range(len(event)):
        labels[np.where(labels == event[k])]=k
        
       #normalise
       #1.calculate the mean 
    for i in range(64):
        num = (data.shape[0]*data.shape[2])
        mean = data[:,i,:].sum()/num
        data[:,i,:]-=mean
        #remove std
        std = np.sqrt((data[:,i,:]**2).sum()/num)
        data[:,i,:]/=std
# =============================================================================
#     data = np.vstack((data[:,:,:150],data[:,:,150:350],data[:,:,300:450]))
# =============================================================================
    data = data[:,:,150:350]
# =============================================================================
#     labels=np.concatenate([labels,labels,labels])
# =============================================================================
        
    #random data
# =============================================================================
#     permutation = np.random.permutation(data.shape[0])
#     data = data[permutation,:,:]
#     labels = labels[permutation]
# =============================================================================
    l = set(labels)
    l = sorted(l)
    mapl = dict()
    for i in range(len(l)):
        mapl[l[i]]=i
    for i in range(len(labels)):
        labels[i]=mapl[labels[i]]
    #reshape
    train_data = np.transpose(data,(0,2,1))
    l = np.zeros(len(labels))
    for i in range(len(labels)):
        l[i]=labels[i]*1.0
    imgs=[]
    tmp =0;
    for i in range(len(labels)-1):
        tmp +=1
        imgs.append(tmp)
        if labels[i+1]!=labels[i]:
            tmp = 0
    imgs = np.array(imgs)
# =============================================================================
#     result = {'data' : train_data, 'labels':l,'features':img}
#     with open('result_raw'+str(sub)+'.pkl','wb') as file:
#         pickle.dump(result,file)
#     labels = img
# =============================================================================
# =============================================================================
# =============================================================================
# #     with open('result.pkl','rb') as file:
# #         result = pickle.load(file)
# #     train_data = result['data']
# #     labels = result['features']
# #     tmp = []
# #     for i in range(labels.shape[0]):
# #         tmp.append(labels[i,0,:])
# #     labels = np.array(tmp)
# #     tmp = []
# #     for i in range(train_data.shape[0]):
# #         tmp.append(train_data[i].flatten())
# #     train_data = np.array(tmp)
# #     print(train_data.shape)
# # =============================================================================
#     with open('result.pkl','rb') as file:
#         result = pickle.load(file)
#     train_data = result['data']
#     labels = result['labels']
#  
#     tmp = []
#     for i in range(train_data.shape[0]):
#         tmp.append(train_data[i].flatten())
#     train_data = np.array(tmp)
#     print(train_data.shape)
#     #get return data
#     return torch.from_numpy(train_data),labels
# =============================================================================
    index = dict()
    index['test'] = np.where(imgs >45 )
    index['valid'] = np.where((40<imgs) & (imgs <=45))
    index['train']= np.where(imgs<=40)
    
    data = dict()
    for split in ('train','valid','test'):
        data[split]=[]
        split_data = train_data[index[split],:,:][0]
        split_label = labels[index[split]]
        
        permutation = np.random.permutation(split_data.shape[0])
        split_data = split_data[permutation,:,:]
        split_label = split_label[permutation]
        for e in range(int(split_data.shape[0]/batch_size)):
            d = dict()
            d['x']=split_data[e*batch_size:(e+1)*batch_size]
            d['y']=split_label[e*batch_size:(e+1)*batch_size]
            data[split].append(d)
            
            
    return data
    
        
        
        
        
        
# =============================================================================
# a = load_data(32)
# =============================================================================

