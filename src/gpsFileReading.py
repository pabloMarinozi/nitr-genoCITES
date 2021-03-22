#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 12:08:36 2021

@author: pablo
"""
import xml.etree.ElementTree as ET
import numpy as np


def asignGPS2Frames(gps_source,n_frames,fps):
    """
    Assign gps coordinates to each frame
    
    Args:
        gps_source (string): a segmented color image
        n_frames (int): number of frames
        fps (int): number of frames per second in the original video

    Returns:
        x (ndarray): x coordinate for each frame
        y (ndarray): y coordinate for each frame
    """
    
    #read gps labels per second from xml file
    tree = ET.parse(gps_source)
    root = tree.getroot()
    gps_info = {}
    for position in root.findall('position'):
        x = float(position.find('x_loc').text)
        y = float(position.find('y_loc').text)
        time = int(position.find('time').text)
        gps_info[time] = (x,y)
        
    #generate arrays with frame info
    frames_indexes = np.arange(n_frames)
    frames_seconds = frames_indexes // int(fps)
    frames_positions_inside_second = frames_indexes % int(fps)
    
    #the first second gps is always wrong in the input videos
    frames_positions_inside_second = frames_positions_inside_second[frames_seconds!=0]
    frames_seconds = frames_seconds[frames_seconds!=0] 

    
    #get base gps coordinate for each second of video
    x_base = np.array([gps_info.get(k)[0] for k in frames_seconds])
    y_base = np.array([gps_info.get(k)[1] for k in frames_seconds])
    
    #get distance traveled during each second (except for the last second)
    last_second = np.max(frames_seconds)
    
    x_distance1 = [abs(gps_info[k][0]-gps_info[k+1][0]) for k in frames_seconds[frames_seconds!=last_second]]
    x_distance2 = [0 for k in frames_seconds[frames_seconds==last_second]]
    x_distance =  x_distance1 + x_distance2
    
    y_distance1 = [abs(gps_info[k][1]-gps_info[k+1][1]) for k in frames_seconds[frames_seconds!=last_second]]
    y_distance2 = [0 for k in frames_seconds[frames_seconds==last_second]]
    y_distance =  y_distance1 + y_distance2

    
    #interpolate gps positions of different frames inside a second 
    x_steps = np.divide(np.array(x_distance), fps, out=np.zeros(np.array(x_distance).shape, dtype=float), where=fps!=0)
    y_steps = np.divide(np.array(y_distance), fps, out=np.zeros(np.array(y_distance).shape, dtype=float), where=fps!=0)
    x = x_base + frames_positions_inside_second*x_steps
    y = y_base + frames_positions_inside_second*y_steps   
    

    return x, y