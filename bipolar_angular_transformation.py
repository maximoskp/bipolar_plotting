#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 22:42:51 2022

@author: max
"""

import sys
if sys.version_info >= (3,8):
    import pickle
else:
    import pickle5 as pickle

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math
from sklearn.decomposition import NMF
# just for loading mnist
import tensorflow as tf
from tensorflow import keras

def NMF_preprocessing(x, d):
    model = NMF(n_components=d, init='random', random_state=0)
    W = model.fit_transform(x)
    H = model.components_
    return W
# end NMF_preprocessing

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return np.array([qx, qy])
# end rotate

def dist_point_line( p, l1, l2 ):
    return np.abs(np.cross(l2-l1, l1-p))/np.linalg.norm(l2-l1)
# d=np.cross(p2-p1,p3-p1)/norm(p2-p1)

def color_mapping(x , i1, i2):
    # three coordinates:
    # c[i,0]: value of color 1, e.g., R value
    # c[i,1]: value of color 2, e.g., B value
    # luminosity can be obtained by (c[i,0]+c[i,1])/2
    # colors are between 0 and 1
    c = np.zeros( (x.shape[0] , 2) )
    for i in range( x.shape[0] ):
        r = max( 0, 1-(np.linalg.norm(x[i,:]-x[i1,:])/max(0.0001, np.linalg.norm(x[i1,:]-x[i2,:])) ) )
        b = max( 0, 1-(np.linalg.norm(x[i,:]-x[i2,:])/max(0.0001, np.linalg.norm(x[i1,:]-x[i2,:])) ) )
        l = max( 0, 1-(dist_point_line( x[i,:] , x[i1,:], x[i2,:] )/max(0.0001, np.linalg.norm(x[i1,:]-x[i2,:])) ) )
        # print('r: ' + str(r) + ' - b: ' + str(b) + ' - l: ' + str(l))
        c[i,0] = l*r
        c[i,1] = l*b
    return c
# end color_mapping

def angle_stretch(x , i1, i2):
    a1 = math.atan2( x[i1,1], x[i1,0] )
    a2 = math.atan2( x[i2,1], x[i2,0] )
    rand_mult = 0.001
    if a1 < a2:
        a1,a2 = a2,a1
    while abs(a1-a2) < 0.001:
        r = np.random.rand( x.shape[0] )
        x[:,1] = x[:,1] + rand_mult*r
        a1 = math.atan2( x[i1,1], x[i1,0] )
        a2 = math.atan2( x[i2,1], x[i2,0] )
        rand_mult += 0.001
    d = np.pi - a1
    # print('i1', i1)
    # print('i2', i2)
    # print('a1', a1)
    # print('a2', a2)
    # print('d', d)
    for i in range( x.shape[0] ):
        a = math.atan2( x[i,1], x[i,0] )
        x[i,:] = rotate( [0,0], x[i,:] , (d+a2)*(a-a2)/(a1-a2) - a2 )
    return x
# end angle_stretch

def differential_plotting(h, i1, i2, k=None, alpha=1.0, colors=True, stretch=True, plot=True):
    # make axis multiplier
    w = np.c_[ h[i1,:] , h[i2,:] ]
    # make reduction
    hh = h@w
    # print('hh[i1,:]', hh[i1,:])
    # print('hh[i2,:]', hh[i2,:])
    if colors:
        c2 = color_mapping(hh, i1, i2)
    else:
        c2 = np.zeros( (hh.shape[0],2) )
    # make rgbcolor
    c = np.insert( c2 , 1, 0, axis=1 )
    if stretch:
        hh = angle_stretch( hh, i1, i2 )
    if plot:
        plt.clf()
        plt.scatter( hh[:,0], hh[:,1], c=c, alpha=0.3)
        plt.scatter(hh[i1,0], hh[i1,1], marker='o', c=[[0,0,0]])
        plt.scatter(hh[i2,0], hh[i2,1], marker='o', c=[[0,0,0]])
        plt.scatter(hh[i1,0], hh[i1,1], marker='x', c=[c[i1,:]])
        plt.scatter(hh[i2,0], hh[i2,1], marker='x', c=[c[i2,:]])
        # plt.plot( hh[:,0], hh[:,1], 'x' );plt.plot(hh[i1,0], hh[i1,1], 'ro');plt.plot(hh[i2,0], hh[i2,1], 'ro', alpha=alpha)
        if k is not None:
            plt.text(hh[i1,0], hh[i1,1], str(k[i1]))
            plt.text(hh[i2,0], hh[i2,1], str(k[i2]))
    return hh,c
# end differential_plotting

def nn_shaping( piece_name1, piece_name2, tonality=True, plot=False, nonnegativity=False, colors=True, stretch=True ):
    datapath = '../nntests_tonefree/data/'
    if tonality:
        datapath = '../nntests/data/'
    with open(datapath + 'states_data.pickle', 'rb') as handle:
        states_data = pickle.load(handle)
    with open(datapath + 'metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)
    # get states combined
    h = np.c_[ np.squeeze( states_data['h_final_np'] ) , np.squeeze( states_data['h_final_np'] ) ]
    if nonnegativity:
        h -= np.min(h)
    # get piece name keys
    k = list( metadata.keys() )
    # get piece indexes
    i1 = k.index( piece_name1 )
    i2 = k.index( piece_name2 )
    hh,c = differential_plotting( h, i1, i2, k=k, alpha=1.0, colors=colors, stretch=True, plot=True )
    return hh,c
# end nn_shaping
