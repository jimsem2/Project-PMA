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

def APPLY_CHORDS(im, spacing=1, axis=1, trim_edges=False, label=True, slice=0, show_slice=True):  # returns ac
    # spacing: space between chords, axis: the axis along which chords are drawn, trim_edges: to remove edges that touch borders
    # label: for chords in same voxel to have same colour
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z
    # show_slice: if True, displays the slice plot

    ac = ps.filters.apply_chords(im=im, spacing=spacing, axis=axis, trim_edges=trim_edges, label=label)
    if show_slice:
        # ps.visualization.imshow() creates its own figure, so we don't pre-create one.
        ps.visualization.imshow((ac), axis=slice, ind=int(im.shape[slice]/2))
        plt.axis(False)
        plt.show()
    return ac


def APPLY_CHORDS_3D(im, spacing=2, trim_edges=True, slice=0, show_slice=True):    # retuns ac3D
    # spacing: space between chords, trim_edges: to remove edges that touch borders
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z
    # show_slice: if True, displays the slice plot

    ac3D = ps.filters.apply_chords_3D(im=im, spacing=spacing, trim_edges=trim_edges)
    ax_chords = 2                                              # which axis chords to show. 1 for x, 2 for y, 3 for z
    if show_slice:
        fig, ax = plt.subplots(1, 2, figsize=[12, 6])
        ax[0].imshow((ac3D + ~im*4)[:, :, int(im.shape[slice]/2)])
        ax[0].axis(False)
        ax[1].imshow(ac3D[:, :, int(im.shape[slice]/2)] == ax_chords)
        ax[1].axis(False)
        plt.show()
    return ac3D


def FILL_CLOSED_PORES(im, conn=26, slice=0, show_slice=True): # returns im_filled_pores
    # conn:  how 'connected' a group of voxels must be. Options: '6', '26'
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z
    # show_slice: if True, displays the slice plot

    im_filled_pores = ps.filters.fill_closed_pores(im, conn=conn)
    if show_slice:
        # ps.visualization.imshow() creates its own figure, so we don't pre-create one.
        ps.visualization.imshow(im_filled_pores, axis=slice, ind=int(im.shape[slice]/2))          # For showing a plane from 3D image, ind: which slice to show
        plt.title(f'Slice along axis {slice}')
        plt.axis(False)
        plt.show()
    return im_filled_pores


def FIND_DISCONNECTED_VOXELS(im, conn=max, show_3d=True): # retunrs discon_voxels
    # conn:  Options: max (when a corner is considered connection between voxels), min (when only face is a connection)
    # show_3d: if True, displays the 3D plot of the disconnected voxels

    discon_voxels = ps.filters.find_disconnected_voxels(im, conn=conn)
    if show_3d:
        fig, ax = plt.subplots(1, figsize=[6, 6])
        x1 = ps.visualization.show_3D(~discon_voxels)
        ax.imshow(x1)
        ax.axis(False)
        plt.show()
    return discon_voxels


def LOCAL_THICKNESS(im, sizes=10, method='dt', divs=2, slice=0, show_slice=True): # returns lt
    # sizes: number of classes to create
    # method: method used for classification, it only affects processing time. Options: 'dt' (default), 'conv', 'bf', 'imj'
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z
    # show_slice: if True, displays the slice plot

    lt = ps.filters.local_thickness(im=im, sizes=sizes, method=method)
    if show_slice:
        # ps.visualization.imshow() creates its own figure, so we don't pre-create one.
        ps.visualization.imshow(lt/im, axis=slice, ind=int(im.shape[slice]/2))
        plt.axis(False)
        plt.show()
    return lt


def POROSIMETRY(im, sizes=15, method='dt', slice=0, show_slice=True, inv_axis=None): # returns pores
    # sizes: Number of spheres of different diameter to be used. Can be a list of values
    # inlets: Characterizing entrance of invading fluid. 0: for the first slice of an axis, -1: for first from the end
    # method: Naturally it sould only affect process speed. Options: 'dt'(DF), 'conv', 'dsi'
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z
    # show_slice: if True, displays the slice plot
    # inv_axis: axis of invasion, 0 for x, 1 for y, 2 for z. If None, no inlets are used

    if inv_axis == 0:
        inlets = np.zeros_like(im)
        inlets[0, :, :] = True
        pores = ps.filters.porosimetry(im=im, sizes=sizes, inlets=inlets, method=method)
    elif inv_axis == 1:
        inlets = np.zeros_like(im)
        inlets[:, 0, :] = True
        pores = ps.filters.porosimetry(im=im, sizes=sizes, inlets=inlets, method=method)
    elif inv_axis == 2:
        inlets = np.zeros_like(im)
        inlets[:, :, 0] = True
        pores = ps.filters.porosimetry(im=im, sizes=sizes, inlets=inlets, method=method)
    else:
        pores = ps.filters.porosimetry(im=im, sizes=sizes, method=method)

    if show_slice:
        # ps.visualization.imshow() creates its own figure, so we don't pre-create one.
        ps.visualization.imshow((pores/im), axis=slice, ind=int(im.shape[slice]/2), interpolation='none')
        plt.axis(False)

    # Creating a cumulative histogram
    fig, ax = plt.subplots(1, figsize=[6, 6])
    ax.hist(pores[im], bins=sizes, cumulative=True, edgecolor='k', density=True, label='access limited')
    ax.set_xlabel('pore radius in voxels')
    ax.set_ylabel('fraction of pore volume smaller than stated size')

    plt.show()
    return pores


def REGION_SIZE(im, slice=0, show_slice=True): # returns RS
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z
    # show_slice: if True, displays the slice plot

    RS = ps.filters.region_size(im=im)
    if show_slice:
        # ps.visualization.imshow() creates its own figure, so we don't pre-create one.
        ps.visualization.imshow((RS), axis=slice, ind=int(im.shape[slice]/2), interpolation='none')
        plt.axis(False)
        plt.show()
    return RS


def TRIM_DISCONNECTED_VOXELS(im, conn='max', slice=0, show_slice=True): # returns trim_discon_voxels
    # conn:  Options: max (when a corner is considered connection between voxels), min (when only face is a connection) 
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z
    # show_slice: if True, displays the slice plot

    inlets = ps.tools.get_border(shape=im.shape, mode='faces')
    trim_discon_voxels = ps.filters.trim_disconnected_voxels(im=im, inlets=inlets, conn=conn)
    if show_slice:
        # ps.visualization.imshow() creates its own figure, so we don't pre-create one.
        ps.visualization.imshow((trim_discon_voxels), axis=slice, ind=int(im.shape[slice]/2), interpolation='none')
        plt.axis(False)
        plt.show()
    return trim_discon_voxels


def TRIM_FLOATING_SOLID(im, conn='max', slice=0, show_slice=True): # returns trimed_im
    # conn:  Options: max (when a corner is considered connection between voxels), min (when only face is a connection) 
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z
    # show_slice: if True, displays the slice plot

    trimed_im = ps.filters.trim_floating_solid(im=im, conn=conn)
    if show_slice:
        # ps.visualization.imshow() creates its own figure, so we don't pre-create one.
        ps.visualization.imshow((trimed_im), axis=slice, ind=int(im.shape[slice]/2), interpolation='none')
        plt.axis(False)
        plt.show()
    return trimed_im


def TRIM_NONPERCOLATING_PATHS(im, perc_axis=0, slice=0, show_slice=True): # returns trimmed_paths
    # perc_axis: axis of percolation
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z
    # show_slice: if True, displays the slice plots

    inlets = np.zeros_like(im)
    outlets = np.zeros_like(im)
    if perc_axis == 0:
        inlets[0, :, :] = True
        outlets[-1, :, :] = True
    elif perc_axis == 1:
        inlets[:, 0, :] = True
        outlets[:, -1, :] = True
    else:
        inlets[:, :, 0] = True
        outlets[:, :, -1] = True

    trimmed_paths = ps.filters.trim_nonpercolating_paths(im=im, inlets=inlets, outlets=outlets)
    if show_slice:
        # ps.visualization.imshow() creates its own figure each call, so we don't pre-create any.
        ps.visualization.imshow((trimmed_paths), axis=slice, ind=int(im.shape[slice]/2), interpolation='none')
        plt.axis(False)
        ps.visualization.imshow((trimmed_paths + (im != trimmed_paths)*0.5), axis=slice, ind=int(im.shape[slice]/2), interpolation='none')
        plt.axis(False)
        plt.show()

    return trimmed_paths
