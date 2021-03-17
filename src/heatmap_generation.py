#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:34:39 2021

@author: pablo
"""
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import cv2

def generate_heatmap(masked_index_image,heatmap_filename):
    """
    Generate a heatmap from an input grayscale image
    
    Args:
        index_image (masked_array): a segmented grayscale image
        heatmap_filename (string): the name of the generated heatmap file
    """
    plt.figure(figsize = (2*7.2,12.8))
    cmap = plt.cm.RdYlGn_r
    cmap.set_bad(color='black')
    norm = cl.Normalize(vmin=0, vmax=masked_index_image.max())
    imgplot = plt.imshow(masked_index_image,cmap=cmap,norm=norm)
    plt.colorbar()
    plt.savefig(heatmap_filename,dpi=100)
    plt.close()
    
def generate_heatmap_video(img_list,size,video_filename):
    """
    Generate a video from heatmap plots
    
    Args:
        img_list (ndarray list): heatmap plots 
        size (tuple): (width, height) of the plots
        video_filename (string): the name of the generated video file
    """    
    out = cv2.VideoWriter(video_filename,cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
    for i in range(len(img_list)):
        out.write(img_list[i])
    out.release()
    print('Heatmap video generated at: ', video_filename)