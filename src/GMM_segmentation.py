#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:23:19 2021

@author: pablo
"""
import os
import cv2
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture as GMM

# error message when image could not be read
IMAGE_NOT_READ = 'IMAGE_NOT_READ'

def read_image(file_path, read_mode=cv2.IMREAD_COLOR):
    """
    Read image file with all preprocessing needed

    Args:
        file_path: absolute file_path of an image file
        read_mode: whether image reading mode is rgb, grayscale or somethin

    Returns:
        np.ndarray of the read image or None if couldn't read

    Raises:
        ValueError if image could not be read with message IMAGE_NOT_READ
    """
    # read image file in grayscale
    image = cv2.imread(file_path, read_mode)

    if image is None:
        raise ValueError(IMAGE_NOT_READ)
    else:
        return image

def segment_leaf_GMM(image_file, leaf_model_file, background_model_file):
    """
    Segments leaf from an image file using gaussian mixtures

    Args:
        image_file (string): full path of an image file
        leaf_model_file (string): full path of the leaf detector model file
        background_model_file (string): full path of the leaf detector model file

    Returns:
        tuple[0] (ndarray): original image to be segmented
        tuple[1] (ndarray): A mask to indicate where leaf is in the image
                            or the segmented image based on marker_intensity value
    """
    if not os.path.isfile(image_file):
        raise ValueError('{}: is not a file'.format(image_file))

    original_image = read_image(image_file)
    imgLUV=cv2.cvtColor(original_image, cv2.COLOR_BGR2LUV) # Trasformamos la imágen al espacio de color  LUV
    imgUV=imgLUV[:,:,1:3]  # le quitamos el L dejando solo UV
    imgUV=imgUV.reshape(((-1,2)))
    with open(leaf_model_file, 'rb') as f:
        leaf_model = pickle.load(f)
    with open(background_model_file, 'rb') as f:
        background_model = pickle.load(f)
    # Obtenemos  la prob de cada pixel a pertenecer a cada componente del GMM  para cada modelo     
    prob_background = background_model.predict_proba(imgUV)
    prob_leaf = leaf_model.predict_proba(imgUV)
    # Compara las probabilidades de cada pixel para cada componente y para cada modelo.
    df_background = np.amax(prob_background,1 ) # elige  la mayor prob en la fila
    df_leaf = np.amax(prob_leaf,1)
    idclases = df_leaf>df_background # En cada pixel compara con si df malezas es más grande que df suelo pone un True si no un False
    mask = idclases.reshape(original_image.shape[0],original_image.shape[1])
    image = original_image.copy()
    image[mask == True] = np.array([0, 0, 0])
    
    return original_image, image