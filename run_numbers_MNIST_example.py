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
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd

import bipolar_angular_transformation as bat

data_folder = 'data/mnist_numbers'
digit1 = 1
digit2 = 8

# %% function declaration

def mnist_example_NMF_save():
    mnist = keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    x_mnist = np.reshape(X_test, (10000, 784))
    nmf_mnist = bat.NMF_preprocessing(x_mnist, 10)
    y_mnist = y_test
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    with open(data_folder + os.sep + 'x_mnist.pickle', 'wb') as handle:
        pickle.dump(x_mnist, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(data_folder + os.sep + 'nmf_mnist.pickle', 'wb') as handle:
        pickle.dump(nmf_mnist, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(data_folder + os.sep + 'y_mnist.pickle', 'wb') as handle:
         pickle.dump(y_mnist, handle, protocol=pickle.HIGHEST_PROTOCOL)
# end mnist_example_save

def mnist_example( digit1, digit2, plot=True, color=True, stretch=True, savedata=True ):
    datapath = data_folder + os.sep
    with open(datapath + 'x_mnist.pickle', 'rb') as handle:
        x_mnist = pickle.load(handle)
    with open(datapath + 'nmf_mnist.pickle', 'rb') as handle:
        nmf_mnist = pickle.load(handle)
    with open(datapath + 'y_mnist.pickle', 'rb') as handle:
        y_mnist = pickle.load(handle)
    # get piece indexes
    where1 = np.where( digit1 == y_mnist )[0]
    i1 = where1[ np.random.randint(0,where1.size) ]
    where2 = np.where( digit2 == y_mnist )[0]
    i2 = where2[ np.random.randint(0,where2.size) ]
    hh, c = bat.differential_plotting( nmf_mnist, i1, i2, k=y_mnist, alpha=1.0, colors=color, stretch=stretch, plot=False )
    z_dimension = (np.linalg.norm(hh[i1,:])/2 + np.linalg.norm(hh[i2,:])/2)*np.mean(c, axis=1)
    if savedata:
        with open( datapath + 'hh_mnist_numbers_example.pickle', 'wb' ) as handle:
            pickle.dump( hh, handle, protocol=pickle.HIGHEST_PROTOCOL )
        with open( datapath + 'c_mnist_numbers_example.pickle', 'wb' ) as handle:
            pickle.dump( c, handle, protocol=pickle.HIGHEST_PROTOCOL )
        with open( datapath + 'z_mnist_numbers_example.pickle', 'wb' ) as handle:
            pickle.dump( z_dimension, handle, protocol=pickle.HIGHEST_PROTOCOL )
        with open( datapath + 'i1i2_mnist_numbers_example.pickle', 'wb' ) as handle:
            pickle.dump( [i1,i2], handle, protocol=pickle.HIGHEST_PROTOCOL )
    # plot
    if plot:
        df = pd.DataFrame({ 'X': hh[:,0],
                            'Y': hh[:,1],
                            'Z': z_dimension,
                            'R': (255.*c[:,0]).astype(int),
                            'G': (255.*c[:,1]).astype(int),
                            'B': (255.*c[:,2]).astype(int)})
        trace = go.Scatter3d(x=df.X,
                      y=df.Y,
                      z=df.Z,
                      mode='markers',
                      marker=dict(size=3,
                                  color=['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(df.R.values, df.G.values, df.B.values)],
                                  opacity=0.5,))
        data = [trace]
        layout = go.Layout(margin=dict(l=0,
                               r=0,
                               b=0,
                               t=0))

        fig = go.Figure(data=data, layout=layout)
        fig.add_trace( go.Scatter3d(x=hh[i1:i1+1,0], y=hh[i1:i1+1,1], z=z_dimension[i1:i1+1], mode='markers', marker=dict(size=10,color=[[255,0,0]],opacity=1)) )
        fig.add_trace( go.Scatter3d(x=hh[i2:i2+1,0], y=hh[i2:i2+1,1], z=z_dimension[i2:i2+1], mode='markers', marker=dict(size=10,color=[[0,0,255]],opacity=1)) )
        fig.update_traces(mode="markers")
        # fig.update_layout(hovermode='x unified')
        fig.show()
        
        # fig = px.scatter_3d(x=hh[:,0], y=hh[:,1], z=z_dimension, color=c, opacity=0.7)
        # fig.show()

        # fig = plt.figure(constrained_layout=True, figsize=(4, 6))
        # gs = GridSpec(nrows=3, ncols=2, figure=fig)
        # ax1 = fig.add_subplot(gs[:2, :])
        # ax1.set_xticks([])
        # ax1.set_yticks([])
        # ax2 = fig.add_subplot(gs[2, 0])
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        # ax3 = fig.add_subplot(gs[2, 1])
        # ax3.set_xticks([])
        # ax3.set_yticks([])
        # ax1.scatter( hh[:,0], hh[:,1], c=c, alpha=0.2)
        # ax1.scatter(hh[i1,0], hh[i1,1], marker='o', c=[[0,0,0]])
        # ax1.scatter(hh[i2,0], hh[i2,1], marker='o', c=[[0,0,0]])
        # ax1.scatter(hh[i1,0], hh[i1,1], marker='x', c=[c[i1,:]])
        # ax1.scatter(hh[i2,0], hh[i2,1], marker='x', c=[c[i2,:]])
        # k = y_mnist
        # ax1.text(hh[i1,0], hh[i1,1], str(k[i1]))
        # ax1.text(hh[i2,0], hh[i2,1], str(k[i2]))
        # ax2.imshow( np.reshape( x_mnist[i1,:] , (28,28) ), cmap='gray_r')
        # ax3.imshow( np.reshape( x_mnist[i2,:] , (28,28) ), cmap='gray_r')
        # plt.show()
        # if savedata:
        #     plt.savefig(datapath + 'mnist_numbers_' + str(digit1) + '-' + str(digit2) + '.png', dpi=500)
    # plt.clf()
    # plt.subplot(3,2,1)
    # plt.plot( hh[:,0], hh[:,1], 'x');plt.plot(hh[i1,0], hh[i1,1], 'ro');plt.plot(hh[i2,0], hh[i2,1], 'ro', alpha=0.5)
    # k = y_mnist
    # if k is not None:
    #     plt.text(hh[i1,0], hh[i1,1], str(k[i1]))
    #     plt.text(hh[i2,0], hh[i2,1], str(k[i2]))
    # plt.subplot(3,2,5)
    # plt.imshow( np.reshape( x_mnist[i1,:] , (28,28) ), cmap='gray_r')
    # plt.subplot(3,2,6)
    # plt.imshow( np.reshape( x_mnist[i2,:] , (28,28) ), cmap='gray_r')
    return hh, c
# end mnist_example

# %% Check if MNIST has been processed with NMF
if not os.path.exists( data_folder + os.sep + 'h_mnist.pickle' ):
    mnist_example_NMF_save()

mnist_example(digit1=digit1, digit2=digit2)

# df = px.data.iris()
# fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
#               color='species')
# fig.show()