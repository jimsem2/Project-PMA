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
np.random.seed(10)

# YELLOW = TRUE = 1 = VOID/PORE // DARK BLUE = FALSE = 0 = SOLID

# Solid Altering Filters
 
def FILL_CLOSED_PORES(im, conn='max', slice=0, show_slice=True): # returns im_filled_pores
    # conn:  how 'connected' a group of voxels must be. Options: 'max' (26-connected), 'min' (6-connected)
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z
    # show_slice: if True, displays the slice plot

    im_filled_pores = ps.filters.fill_closed_pores(im, conn=conn)
    if show_slice:
        # ps.visualization.imshow() creates its own figure, so we don't pre-create one.
        ps.visualization.imshow(im_filled_pores, axis=slice, ind=int(im.shape[slice]/2))
        plt.gcf().canvas.manager.set_window_title("FILL CLOSED PORES")
        plt.title(f'Slice along axis {slice}')
        plt.axis(False)
        plt.show(block=False)
    return im_filled_pores


def TRIM_DISCONNECTED_VOXELS(im, conn='max', inlet_faces='all', slice=0, show_slice=True): # returns trim_discon_voxels
    # conn:        Options: max (when a corner is considered connection between voxels), min (when only face is a connection)
    # inlet_faces: which faces act as inlets. Options: 'all' (default),'x0' first plane along x,'x1' last plane along x, 
    #               'y0', 'y1', 'z0', 'z1', list of the above strings, e.g. ['x0', 'x1']
    # slice:       according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z
    # show_slice:  if True, displays the slice plot

    if inlet_faces == 'all':
        inlets = ps.generators.borders(shape=im.shape, mode="faces")
    else:
        if isinstance(inlet_faces, str):
            inlet_faces = [inlet_faces]
        inlets = np.zeros(im.shape, dtype=bool)
        _s = np.s_[:]   # "select all" sentinel
        face_map = {
            'x0': (0,  _s, _s),
            'x1': (-1, _s, _s),
            'y0': (_s, 0,  _s),
            'y1': (_s, -1, _s),
            'z0': (_s, _s, 0),
            'z1': (_s, _s, -1),
        }
        for face in inlet_faces:
            if face not in face_map:
                raise ValueError(f"Unknown inlet_face '{face}'. Choose from: 'all', 'x0', 'x1', 'y0', 'y1', 'z0', 'z1'.")
            inlets[face_map[face]] = True

    trim_discon_voxels = ps.filters.trim_disconnected_voxels(im=im, inlets=inlets, conn=conn)
    if show_slice:
        # ps.visualization.imshow() creates its own figure, so we don't pre-create one.
        ps.visualization.imshow((trim_discon_voxels), axis=slice, ind=int(im.shape[slice]/2), interpolation='none')
        plt.gcf().canvas.manager.set_window_title("TRIM DISCONNECTED VOXELS")
        plt.title("TRIM DISCONNECTED VOXELS")
        plt.axis(False)
        plt.show(block=False)
    return trim_discon_voxels


def TRIM_FLOATING_SOLID(im, conn='max', slice=0, show_slice=True): # returns trimed_im
    # conn:  Options: max (when a corner is considered connection between voxels), min (when only face is a connection) 
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z
    # show_slice: if True, displays the slice plot

    trimed_im = ps.filters.trim_floating_solid(im=im, conn=conn)
    if show_slice:
        # ps.visualization.imshow() creates its own figure, so we don't pre-create one.
        ps.visualization.imshow((trimed_im), axis=slice, ind=int(im.shape[slice]/2), interpolation='none')
        plt.gcf().canvas.manager.set_window_title("TRIM FLOATING SOLID")
        plt.title("TRIM FLOATING SOLID")
        plt.axis(False)
        plt.show(block=False)
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
        plt.gcf().canvas.manager.set_window_title("TRIM NONPERCOLATING PATHS")
        plt.title("TRIM NONPERCOLATING PATHS")
        plt.axis(False)
        ps.visualization.imshow((trimmed_paths + (im != trimmed_paths)*0.5), axis=slice, ind=int(im.shape[slice]/2), interpolation='none')
        plt.gcf().canvas.manager.set_window_title("TRIM NONPERCOLATING PATHS - Comparison")
        plt.title("TRIM NONPERCOLATING PATHS - Comparison")
        plt.axis(False)
        plt.show(block=False)

    return trimmed_paths


def CROP_CENTER(im, size=300, slice=0, show_slice=True):
    # size: side length of the output cube (voxels); each axis is cropped symmetrically
    # slice: axis for the 2D slice preview (0=x, 1=y, 2=z)
    # show_slice: if True, display the centre slice of the cropped result
    cropped_parts = []
    for s in im.shape:
        start = max(0, (s - size) // 2)
        end = start + min(s, size)
        cropped_parts.append((start, end))
    (x0, x1), (y0, y1), (z0, z1) = cropped_parts
    result = im[x0:x1, y0:y1, z0:z1]
    print(f"Cropped from {im.shape} to {result.shape}.")
    if show_slice:
        ps.visualization.imshow(result, axis=slice, ind=int(result.shape[slice] / 2))
        plt.gcf().canvas.manager.set_window_title("CROP CENTER")
        plt.title(f"Cropped: {result.shape}")
        plt.axis(False)
        plt.show(block=False)
    return result


# Non-Solid Altering Filters

def APPLY_CHORDS(im, spacing=1, axis=1, trim_edges=False, label=True, slice=0, show_slice=True):  # returns ac
    # spacing: space between chords
    # axis: the axis along which chords are drawn
    # trim_edges: when True, removes edges that touch borders
    # label: when True, for chords in same voxel to have same colour
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z
    # show_slice: if True, displays the slice plot

    ac = ps.filters.apply_chords(im=im, spacing=spacing, axis=axis, trim_edges=trim_edges, label=label)
    if show_slice:
        # ps.visualization.imshow() creates its own figure, so we don't pre-create one.
        ps.visualization.imshow((ac), axis=slice, ind=int(im.shape[slice]/2))
        plt.gcf().canvas.manager.set_window_title("APPLY CHORDS")
        plt.title("APPLY CHORDS")
        plt.axis(False)
        plt.show(block=False)
    return ac


def APPLY_CHORDS_3D(im, spacing=2, trim_edges=True, slice=0, show_slice=True):    # retuns ac3D
    # spacing: space between chords
    # trim_edges: when True, removes edges that touch borders
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z
    # show_slice: if True, displays the slice plot

    ac3D = ps.filters.apply_chords_3D(im=im, spacing=spacing, trim_edges=trim_edges)
    ax_chords = 2                                              # which axis chords to show. 1 for x, 2 for y, 3 for z
    if show_slice:
        fig, ax = plt.subplots(1, 2, figsize=[12, 6])
        fig.canvas.manager.set_window_title("APPLY CHORDS 3D")
        ax[0].imshow((ac3D + ~im*4)[:, :, int(im.shape[slice]/2)])
        ax[0].set_title("APPLY CHORDS 3D")
        ax[0].axis(False)
        ax[1].imshow(ac3D[:, :, int(im.shape[slice]/2)] == ax_chords)
        ax[1].set_title("APPLY CHORDS 3D - Selected Axis")
        ax[1].axis(False)
        plt.show(block=False)
    return ac3D


def FIND_DISCONNECTED_VOXELS(im, conn=max, show_3d=True): # retunrs discon_voxels
    # conn:  Options: max (when a corner is considered connection between voxels), min (when only face is a connection)
    # show_3d: if True, displays the 3D plot of the disconnected voxels

    discon_voxels = ps.filters.find_disconnected_voxels(im, conn=conn)
    if show_3d:
        fig, ax = plt.subplots(1, figsize=[6, 6])
        fig.canvas.manager.set_window_title("FIND DISCONNECTED VOXELS")
        x1 = ps.visualization.show_3D(~discon_voxels)
        ax.imshow(x1)
        ax.set_title("FIND DISCONNECTED VOXELS")
        ax.axis(False)
        plt.show(block=False)
    return discon_voxels


def LOCAL_THICKNESS(im, sizes=10, method='dt', slice=0, show_slice=True): # returns lt
    # sizes: number of classes to create
    # method: method used for classification, it only affects processing time. Options: 'dt' (default), 'conv', 'bf', 'imj'
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z
    # show_slice: if True, displays the slice plot

    lt = ps.filters.local_thickness(im=im, sizes=sizes, method=method)
    if show_slice:
        # ps.visualization.imshow() creates its own figure, so we don't pre-create one.
        ps.visualization.imshow(lt/im, axis=slice, ind=int(im.shape[slice]/2))
        plt.gcf().canvas.manager.set_window_title("LOCAL THICKNESS")
        plt.title("LOCAL THICKNESS")
        plt.axis(False)
        plt.show(block=False)
    return lt


def POROSIMETRY(im, sizes=15, method='dt', inv_axis=None, slice=0, show_slice=True): # returns prsm
    # sizes: Number of spheres of different diameter to be used. Can be a list of values
    # method: Naturally it sould only affect process speed. Options: 'dt'(DF), 'conv', 'dsi'
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z
    # show_slice: if True, displays the slice plot
    # inv_axis: axis of invasion, 0 for x, 1 for y, 2 for z. If None, no inlets are used

    if inv_axis == 0:
        inlets = np.zeros_like(im)
        inlets[0, :, :] = True
        prsm = ps.filters.porosimetry(im=im, sizes=sizes, inlets=inlets, method=method)
    elif inv_axis == 1:
        inlets = np.zeros_like(im)
        inlets[:, 0, :] = True
        prsm = ps.filters.porosimetry(im=im, sizes=sizes, inlets=inlets, method=method)
    elif inv_axis == 2:
        inlets = np.zeros_like(im)
        inlets[:, :, 0] = True
        prsm = ps.filters.porosimetry(im=im, sizes=sizes, inlets=inlets, method=method)
    else:
        prsm = ps.filters.porosimetry(im=im, sizes=sizes, method=method)

    if show_slice:
        ps.visualization.imshow((prsm/im), axis=slice, ind=int(im.shape[slice]/2), interpolation='none')
        plt.gcf().canvas.manager.set_window_title("POROSIMETRY")
        plt.title("POROSIMETRY")
        plt.axis(False)

    # Creating a cumulative histogram
    fig, ax = plt.subplots(1, figsize=[6, 6])
    fig.canvas.manager.set_window_title("POROSIMETRY - Histogram")
    ax.hist(prsm[im], bins=sizes, cumulative=True, edgecolor='k', density=True, label='access limited')
    ax.set_title("POROSIMETRY")
    ax.set_xlabel('Pore Radius In Voxels')
    ax.set_ylabel('Fraction Of Pore Volume Smaller Than Stated Size')

    plt.show(block=False)
    return prsm


def REGION_SIZE(im, slice=0, show_slice=True): # returns RS
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z
    # show_slice: if True, displays the slice plot

    RS = ps.filters.region_size(im=im)
    if show_slice:
        ind = int(im.shape[slice] / 2)
        if slice == 0:
            plane = RS[ind, :, :]
        elif slice == 1:
            plane = RS[:, ind, :]
        else:
            plane = RS[:, :, ind]
        fig, ax = plt.subplots(figsize=[6, 6])
        fig.canvas.manager.set_window_title("REGION SIZE")
        ax.imshow(plane)
        ax.set_title("REGION SIZE")
        ax.axis(False)
        plt.show(block=False)
    return RS

def SNOW2_REGIONS(im, voxel_size=1, slice=0, show_slice=True): # returns regions
    # voxel_size: physical length of each voxel (m) — passed to snow2 for network geometry
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z
    # show_slice: if True, displays the slice plot
        
    snow2 = ps.networks.snow2(phases=im, voxel_size=voxel_size, boundary_width=0)
    regions = snow2.regions
    if show_slice:
        ind = int(im.shape[slice] / 2)
        if slice == 0:
            plane = regions[ind, :, :]
        elif slice == 1:
            plane = regions[:, ind, :]
        else:
            plane = regions[:, :, ind]
        fig, ax = plt.subplots(figsize=[6, 6])
        fig.canvas.manager.set_window_title("SNOW2 REGIONS")
        ax.imshow(plane)
        ax.set_title("SNOW2 REGIONS")
        ax.axis(False)
        plt.show(block=False)

    return regions

