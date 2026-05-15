import porespy as ps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as spst
import scipy.ndimage as spim
import skimage
from skimage.morphology import  ball, cube, binary_dilation
import time
from porespy.tools import randomize_colors
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

# YELLOW = TRUE = 1 = VOID/PORE // DARK BLUE = FALSE = 0 = SOLID

def APPLY_CHORDS(im, spacing=1, axis=1, trim_edges=False, label=True):  # returns x
    # spacing: space between chords, axis: the axis along which chords are drawn, trim_edges: to remove edges that touch borders
    # label: for chords in same voxel to have same colour

    x = ps.filters.apply_chords(im=im, spacing=spacing, axis=axis, trim_edges=trim_edges, label=label)
    fig, ax = plt.subplots(1, figsize=[4, 4])
    slice = 1 
    ps.visualization.imshow((x), axis=slice, ind=int(im.shape[slice]/2))
    ax.axis(False)

    plt.show()
    return x

def APPLY_CHORDS_3D(im, spacing=2, trim_edges=True):    # retuns chords
    # spacing: space between chords, trim_edges: to remove edges that touch borders

    chords = ps.filters.apply_chords_3D(im=im, spacing=spacing, trim_edges=trim_edges)

    slice = 2                                                  # according to which slice to show the plot 0=yz, 1=xz, 2=xy
    ax_chords = 2                                              # which axis chords to show. 1 for x, 2 for y, 3 for z
    fig, ax = plt.subplots(1, 2, figsize=[12, 6])
    ax[0].imshow((chords + ~im*4)[:, :, int(im.shape[slice]/2)])
    ax[0].axis(False)
    ax[1].imshow(chords[:, :, int(im.shape[slice]/2)] == ax_chords)
    ax[1].axis(False)

    plt.show()
    return chords

def FILL_CLOSED_PORES(im, conn=26): # returns im_filled UPDATE PORESPY
    # conn:  how ‘connected’ a group of voxels must be. Options: '6', '26'

    im_filled = ps.filters.fill_closed_pores(im, conn=conn)
    fig, ax = plt.subplots(1, figsize=[6, 6])
    slice = 1                                                                           # according to which slice to show the plot 0=yz, 1=xz, 2=xy
    ps.visualization.imshow(im_filled, axis=slice, ind=int(im.shape[slice]/2))          # For showing a plane from 3D image, ind: which slice to show
    ax.set_title(f'xz slice')
    ax.axis(False)

    plt.show()
    return im_filled

def FIND_DISCONNECTED_VOXELS(im, conn=max): # retunrs discon_voxels
    # conn:  Options: max (when a corner is considered connection between voxels), min (when only face is a connection) 

    discon_voxels = ps.filters.find_disconnected_voxels(im, conn=conn)
    fig, ax = plt.subplots(1, figsize=[6, 6])
    x1 = ps.visualization.show_3D(~discon_voxels)
    ax.imshow(x1)
    ax.axis(False)

    plt.show()
    return discon_voxels

def LOCAL_THICKNESS(im, sizes=10, method='dt', divs=2): # returns lt
    # sizes: number of classes to create
    # method: method used for classification, it only affects processing time. Options: 'dt' (default), 'conv', 'bf', 'imj'

    lt = ps.filters.local_thickness(im=im, sizes=sizes, method=method, divs=divs)
    slice = 1                                                   # according to which slice to show the plot 0=yz, 1=xz, 2=xy
    fig, ax = plt.subplots(1, figsize=[6, 6])
    ps.visualization.imshow(lt/im, axis=slice, ind=int(im.shape[slice]/2))
    ax.axis(False)

    plt.show()
    return lt

def POROSIMETRY(im, sizes=15, method='dt'): # returns pores
    # sizes: Number of spheres of different diameter to be used. Can be a list of values
    # inlets: Characterizing entrance of invading fluid. 0: for the first slice of an axis, -1: for first from the end
    # method: Naturally it sould only affect process speed. Options: 'dt'(DF), 'conv', 'dsi'

    fluid_entr = input('\nWhould you like the invading fluid to enter only through one side (y/n): ')
    if fluid_entr == 'y':
        inlets = np.zeros_like(im)
        inlets[0, :, :] = True                                  # [x,:,:]: the fluid will enter through a x axis side
        pores = ps.filters.porosimetry(im=im, sizes=sizes, inlets=inlets, method=method)
    else:
        pores = ps.filters.porosimetry(im=im, sizes=sizes, method=method)

    fig, ax = plt.subplots(1, figsize=[6, 6])
    slice = 1                                                   # according to which slice to show the plot 0=yz, 1=xz, 2=xy
    ps.visualization.imshow((pores/im), axis=slice, ind=int(im.shape[slice]/2), interpolation='none')
    ax.axis(False)

    # Creating a cumulative histogram
    fig, ax = plt.subplots(1, figsize=[6, 6])
    ax.hist(pores[im], bins=sizes, cumulative=True, edgecolor='k', density=True, label='access limited')
    ax.set_xlabel('pore radius in voxels')
    ax.set_ylabel('fraction of pore volume smaller than stated size')

    plt.show()
    return pores

def REGION_SIZE(im): # returns x
    x = ps.filters.region_size(im=im)

    slice = 0                                                   # according to which slice to show the plot 0=yz, 1=xz, 2=xy
    ind=int(im.shape[slice]/2)
    fig, ax = plt.subplots(1, figsize=[6, 6])
    ps.visualization.imshow((x), axis=slice, ind=int(im.shape[slice]/2), interpolation='none')
    plt.axis(False)

    plt.show()
    return x


    # im: optional, to differentiate solid and uninvaded voxels
    # voxel_size: resolution of  image in meters per voxel edge, g: gravitational constant

    from edt import edt
    drn = ps.simulations.drainage(im=im, voxel_size=1, g=0)
    seq = ps.filters.satn_to_seq(satn=drn.im_satn, im=im)

    slice = 0                                                   # according to which slice to show the plot 0=yz, 1=xz, 2=xy
    fig, ax = plt.subplots(1, figsize=[6, 6])
    ps.visualization.imshow((drn.im_satn/im), axis=slice, ind=int(im.shape[slice]/2), interpolation='none')
    plt.axis(False)
    plt.title('Invasion map by saturation')
    fig, ax = plt.subplots(1, figsize=[6, 6])
    ps.visualization.imshow((seq/im), axis=slice, ind=int(im.shape[slice]/2), interpolation='none')
    plt.axis(False)
    plt.title('Invasion map by sequence')

    plt.show()
    return seq

def TRIM_DISCONNECTED_VOXELS(im, conn='max'): # returns im2
    # conn:  Options: max (when a corner is considered connection between voxels), min (when only face is a connection) 

    inlets = ps.tools.get_border(shape=im.shape, mode='faces')
    im2 = ps.filters.trim_disconnected_voxels(im=im, inlets=inlets, conn=conn)

    fig, ax = plt.subplots(1, figsize=[6, 6])
    slice = 1                                                   # according to which slice to show the plot 0=yz, 1=xz, 2=xy
    ps.visualization.imshow((im2), axis=slice, ind=int(im.shape[slice]/2), interpolation='none')
    ax.axis(False)

    plt.show()
    return im2

def TRIM_FLOATING_SOLID(im, conn='max'): # returns x
    # conn:  Options: max (when a corner is considered connection between voxels), min (when only face is a connection) 

    x = ps.filters.trim_floating_solid(im=im, conn=conn)

    fig, ax = plt.subplots(1, figsize=[6, 6])
    slice = 1                                                   # according to which slice to show the plot 0=yz, 1=xz, 2=xy
    ps.visualization.imshow((x), axis=slice, ind=int(im.shape[slice]/2), interpolation='none')
    ax.axis(False)

    plt.show()
    return x

def TRIM_NONPERCOLATING_PATHS(im, perc_axis='x'):
    # perc_axis: axis of percolation

    inlets = np.zeros_like(im)
    outlets = np.zeros_like(im)
    if perc_axis == 'x':
        inlets[0, :, :] = True
        outlets[-1, :, :] = True
    elif perc_axis == 'y':
        inlets[:, 0, :] = True
        outlets[:, -1, :] = True
    else:
        inlets[:, :, 0] = True
        outlets[:, :, -1] = True

    x = ps.filters.trim_nonpercolating_paths(im=im, inlets=inlets, outlets=outlets)

    slice = 1                                                   # according to which slice to show the plot 0=yz, 1=xz, 2=xy
    fig, ax = plt.subplots(1, figsize=[6, 6])
    ps.visualization.imshow((x), axis=slice, ind=int(im.shape[slice]/2), interpolation='none')
    ax.axis(False)
    fig, ax = plt.subplots(1, figsize=[6, 6])
    ps.visualization.imshow((x + (im != x)*0.5), axis=slice, ind=int(im.shape[slice]/2), interpolation='none')
    ax.axis(False)
    plt.show()

    return x

# TRIM_NONPERCOLATING_PATHS(im, perc_axis='z')
