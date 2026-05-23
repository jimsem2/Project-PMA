import time
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import porespy as ps
import scipy as sp
import scipy.ndimage as spim
import scipy.stats as spst
import skimage
# from porespy.tools import randomize_colors
from skimage.morphology import ball, binary_dilation, cube

# Networks Libraries
import sklearn
# import tensorflow as tf
import importlib
import inspect
import warnings
import h5py
from sklearn.metrics import r2_score
import os
import openpnm as op

# YELLOW = TRUE = 1 = VOID/PORE // DARK BLUE = FALSE = 0 = SOLID

np.random.seed(10)


length = 100
ind = int(length/2)
im = ps.generators.blobs(shape=[length, length, length], porosity=0.7)

def MAP_TO_REGIONS(im, slice=0):
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z
    ws = op.Workspace()
    ws.settings['loglevel'] = 50
    ps.visualization.set_mpl_style()

    # Creating network
    fig, ax = plt.subplots(1, figsize=[4, 4])
    ps.visualization.imshow((im), axis=slice, ind=int(im.shape[slice]/2))
    snow_output = ps.networks.snow2(im, voxel_size=1)
    pn = op.io.network_from_porespy(snow_output.network)

    # Assigning values to network
    pn['pore.values'] = np.random.rand(pn.Np)
    # Assigning the values to the image
    fig, ax = plt.subplots()
    reg = ps.networks.map_to_regions(regions=snow_output.regions.T, values=pn['pore.values'])
    plt.imshow((reg)[:, :, ind])
    plt.show()


def REGIONS_TO_NETWORK(im, slice=0):
    # phases: image indicating to which phase each voxel belongs
    # voxel_size: resolution of image
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z

    snow = ps.filters.snow_partitioning(im)
    fig, ax = plt.subplots(1, figsize=[4, 4])
    ps.visualization.imshow((snow.regions/im), axis=slice, ind=int(im.shape[slice]/2))

    # Printing network information
    net1 = ps.networks.regions_to_network(regions=snow.regions)
    for item in net1.keys():
        print(item)

    # For multiphased objects
    snow2 = ps.filters.snow_partitioning(~im)
    fig, ax = plt.subplots(1, figsize=[4, 4])
    ps.visualization.imshow((snow2.regions), axis=slice, ind=int(im.shape[slice]/2))
    ws = snow.regions + (snow2.regions + snow.regions.max())*~im
    fig, ax = plt.subplots(1, figsize=[4, 4])
    ps.visualization.imshow((ws), axis=slice, ind=int(im.shape[slice]/2))

    # Printing network information
    net2 = ps.networks.regions_to_network(regions=ws, phases=im.astype(int)+1, voxel_size=1e-5)
    for item in net2.keys():
        print(item)
    plt.show()


def SNOW2(im, slice=0):
    # boundary_width: Adding pores/nodes on the boundaries of images is helpful when conducting transport simulations
    # voxel_size: resolution of image
    # parallelization: dividing the image into chunks and processing each in parallel (more info on porespy.filters.snow_partitioning_parallel)
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z

    fig, ax = plt.subplots(1, figsize=[4, 4])
    ps.visualization.imshow((im), axis=slice, ind=int(im.shape[slice]/2))

    # Applying snow2 filter
    net2 = ps.networks.snow2(phases=im.astype(int) + 1, boundary_width=[[2, 0], 4, [0, 2]], parallelization={'cores': 1, 'divs': 2, 'overlap': 1})   # adding '.astype(int)+1 to make it two phased
    fig, ax = plt.subplots(1, figsize=[4, 4])
    ps.visualization.imshow((net2.regions), axis=slice, ind=int(im.shape[slice]/2))

    # Overlaying the network with the image
    pn2 = op.io.network_from_porespy(net2.network)
    #   Differnieting connections and coordinates of the 2 phases
    ax = op.visualization.plot_connections(pn2, throats=pn2['throat.phase2_phase2'], c='k')
    ax = op.visualization.plot_connections(pn2, throats=pn2['throat.phase2_phase1'], c='grey', ax=ax)
    ax = op.visualization.plot_coordinates(pn2, pores=pn2['pore.phase2'], c='r', markersize=50, ax=ax)
    ax = op.visualization.plot_coordinates(pn2, pores=pn2['pore.phase1'], c='g', markersize=50, ax=ax)
    fig, ax = plt.gcf(), plt.gca()      # gcf: get current figure, gca: get current axis
    ax.imshow((net2.regions.T)[:, :, ind])
    ax.axis(False)

    # Using Alias
    net2alias = ps.networks.snow2(phases=im.astype(int) + 1, phase_alias={1: 'solid', 2: 'void'}, boundary_width=0)
    pn2alias = op.io.network_from_porespy(net2alias.network)
    print(pn2alias)
    
    plt.show()
