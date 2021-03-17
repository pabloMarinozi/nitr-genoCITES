#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:32:51 2021

@author: pablo
"""
import cv2
import numpy as np

def get_index_image(output_image, index):
    """
    Assign to each pixel an index based on its rgb value
    
    Args:
        output_image (ndarray): a segmented color image
        index (string): name of the index to be calculated

    Returns:
        (ndarray): a grayscale image where each pixel has an index value
    """
    B, G, R = cv2.split(output_image)
    B, G, R = B.astype(int), G.astype(int),R.astype(int)
    total = B + G + R
    
    if index == "r":
        return np.divide(R, total, out=np.zeros(R.shape, dtype=float), where=total!=0)
    elif index == "g":
        return np.divide(G, total, out=np.zeros(R.shape, dtype=float), where=total!=0)
    elif index == "b":
        return np.divide(B, total, out=np.zeros(R.shape, dtype=float), where=total!=0)
    elif index == "rb":
        return np.divide(R, B, out=np.zeros(R.shape, dtype=float), where=B!=0)
    elif index == "rg":
        return np.divide(R, G, out=np.zeros(R.shape, dtype=float), where=G!=0)
    elif index == "bg":
        return np.divide(B, G, out=np.zeros(R.shape, dtype=float), where=G!=0)