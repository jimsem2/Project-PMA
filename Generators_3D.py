import porespy as ps
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats as spst
import pandas as pd
ps.visualization.set_mpl_style()
np.random.seed(10)                      # eraze for randomly creating an object

# YELLOW/TRUE = VOID/PORE, DARK BLUE/FALSE = SOLID

def BLOBS(shape, porosity=0.5, blobiness=1):  #returns im
    # When: pososity=None -> return: greyscale image
    # Interpolation is used for adding smoothness. im<0.5 converts greyscale to boolean
    # blobiness: integer/list, higher number -> more blobs

    im = ps.generators.blobs(shape=shape, porosity=porosity, blobiness=blobiness)
    # ax = plt.figure().add_subplot(projection='3d')                                  # For maneuverable plot !! takes a lot of time
    # ax.voxels(im, edgecolor='none', linewidth=0.25)
    fig, ax = plt.subplots(1, 1, figsize=[6, 6])                                    # For static plot
    ax.imshow(ps.visualization.show_3D(im), interpolation='none')

    plt.show()
    return im

def CYLINDERS(shape, radius=2, ncylinders=0, porosity=0, cyl_length=None):   #returns im, ! Only 3D shapes
    # By controlling the number of cyliders
    # ncylinders: number of cyl
    if ncylinders > 0:
        im = ps.generators.cylinders(shape=shape, r = radius, ncylinders = ncylinders, length = cyl_length)

    # By controlling the porosity
    # cyl_length: cylinder length
    else:
        im = ps.generators.cylinders(shape=shape, r = radius, porosity = porosity, length = cyl_length)
        print(f'Actual porosity is {im.sum()/im.size}')

    fig, ax = plt.subplots(1, 1, figsize=[6, 6])  
    ax.imshow(ps.visualization.show_3D(im), interpolation='none')
    # ax = plt.figure().add_subplot(projection='3d')                  # this plot takes a lot of time for big shapes
    # ax.voxels(~im, edgecolor='k', linewidth=0.25)

    plt.show()
    return im


    # shape: list, inlet, outlet: 0 for x axis, 1 for y, 2 for z, the values of the faces becomes TRUE

    faces = ps.generators.faces(shape=shape, inlet=inlet, outlet=outlet)
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(faces, edgecolor='k', linewidth=0.25)

    plt.show()
    return faces

def OVERLAPPING_SPHERES(shape, radius, porosity, maxiter=10):  #returns im
    # r: controls size of spheres, maxiter: number of of iterations used to match the desired porosity default=10

    im = ps.generators.overlapping_spheres(shape=shape, r = radius, porosity=porosity, maxiter=maxiter)
    fig, ax = plt.subplots(1, 1, figsize=[4, 4])
    ax.imshow(ps.visualization.show_3D(im), interpolation='none')
    ax.axis(False)
    print('porosity = ', ps.metrics.porosity(im))

    plt.show()
    return im

def POLYDISPERSE_SPHERES(shape, porosity=0.5, r_min=4, nbins=3):  #returns im
    # dist: statistical distribution of spheres diameters
    # nbins: number of discrete sphere sizes (default=5), r_min: smallest sphere to generate (default=5)

    dist = spst.norm(loc=10, scale=5)
    im = ps.generators.polydisperse_spheres(shape=shape, porosity = porosity, nbins=nbins, r_min=r_min, dist=dist)
    print(f'porosity = {im.sum()/im.size}')
    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    ax.imshow(ps.visualization.show_3D(im), interpolation='none')
    ax.axis(False)

    plt.show()
    return im

def PSEUDO_GRAVITY_PACKING(im, radius, maxiter, clearance=0, pk_axis=0):  #returns packing
    # im: can also be a list
    # r: radius of spheres, # clearance: <0 for overlaping between spheres, # maxiter: maximum number of spheres to add
    # axis: direction of gravity (default axis=0 -> x=0)

    if isinstance(im, list):
        im = np.zeros(im, dtype=bool)
        packing = ps.generators.pseudo_gravity_packing(im=~im, r=radius, maxiter=maxiter, clearance=clearance, axis=pk_axis)
        fig, ax = plt.subplots(1, figsize=[8, 8])
        ax.imshow(ps.visualization.show_3D(~packing), interpolation='none', origin='lower')
        ax.axis(False)
    else:
        packing = ~ps.generators.pseudo_gravity_packing(im=im, r=radius, maxiter=maxiter, clearance=clearance, axis=pk_axis)
        fig, ax = plt.subplots(1, figsize=[8, 8])
        ax.imshow(ps.visualization.show_3D(packing), interpolation='none', origin='lower')
        ax.axis(False)
    plt.show()
    return packing

def VORONOI_EDGES(shape, thickness, ncells, flat_faces = True):
    # ncells: number of cells # flat_faces = false : for edges to be true, r: thickness of edge (default=1)

    im = ps.generators.voronoi_edges(shape=shape, r=thickness, ncells=ncells, flat_faces=flat_faces)
    fig, ax = plt.subplots(figsize=[8, 8])
    ax.imshow(ps.visualization.show_3D(~im), interpolation='none')          # remove ~ to make edges solid
    ax.axis(False)

    plt.show()
    return im

