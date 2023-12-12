import math, pickle
import numpy as np
np.bool = np.bool_

import matplotlib.pyplot as plt
import tensorflow as tf

import os
from pathlib import Path
import sys
import collections
import wget
import git
from tqdm import tqdm
import tarfile

from colorama import Fore, Back
from colorama import init
init(autoreset=True)
import colored_traceback
colored_traceback.add_hook(always=True)
import PIL.Image
import cv2
from utils import cprintf


def save(dataset, file):
    
    with open(file, 'wb') as fo:
        
        pickle.dump(dataset, fo)


def unpickle(file):
    
    with open(file, 'rb') as fo:
        
        dict = pickle.load(fo, encoding='bytes')
    
    return dict


def batch_run(function, images, batch_size=5000):
    '''
    function   : lambda function taking images with shape [N,H,W,C] as input
    images     : tensor of shape [N,H,W,C]
    batch_size : batch size
    '''
    
    res = []
    
    for i in range(math.ceil(len(images) / batch_size)):
        
        res.append(function(images[i*batch_size:(i+1)*batch_size]))
    
    return np.concatenate(res, axis=0)


def preprocess(attributions, q1, q2, use_abs=False):
    
    if use_abs:
        attributions = np.abs(attributions)
    
    if tf.is_tensor(attributions):
        attributions = attributions.numpy()
        # attributions.eval(session=tf.compat.v1.Session())
    attributions = np.sum(attributions, axis=-1)
    if attributions.ndim == 2:
        attributions = attributions[np.newaxis is None,:,:]
    cprintf(f'{attributions.shape}', 'l_blue')
    a_min = np.percentile(attributions, q1, axis=(1,2), keepdims=True)
    a_max = np.percentile(attributions, q2, axis=(1,2), keepdims=True)
    
    pos = np.tile(a_min > 0, [1,attributions.shape[1],attributions.shape[2]])
    ind = np.where(attributions < a_min)
    
    attributions = np.clip(attributions, a_min, a_max)
    attributions[ind] = (1 - pos[ind]) * attributions[ind]
    
    return attributions


def pixel_range(img):
    vmin, vmax = np.min(img), np.max(img)

    if vmin * vmax >= 0:
        
        v = np.maximum(np.abs(vmin), np.abs(vmax))
        
        return [-v, v], 'bwr'
    
    else:

        if -vmin > vmax:
            vmax = -vmin
        else:
            vmin = -vmax

        return [vmin, vmax], 'bwr'


def scale(x):
    
    return x / 127.5 - 1.0

def extract_all_files(tar_file_path, extract_to):
    with tarfile.open(tar_file_path, 'r') as tar:
        tar.extractall(extract_to)