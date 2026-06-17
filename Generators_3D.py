from turtle import shape

import porespy as ps
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats as spst
import pandas as pd
import imageio
ps.visualization.set_mpl_style()
np.random.seed(10)                      # eraze for randomly creating an object

# YELLOW/TRUE = VOID/PORE, DARK BLUE/FALSE = SOLID

def BLOBS(shape, porosity=0.5, blobiness=1, invert=False):  #returns im
    # When: pososity=None -> return: greyscale image
    # Interpolation is used for adding smoothness. im<0.5 converts greyscale to boolean
    # blobiness: integer/list, higher number -> more blobs
    # invert: if True, inverts the image (pores become solid and vice versa)

    im = ps.generators.blobs(shape=shape, porosity=porosity, blobiness=blobiness)
    if invert:
        im = ~im
    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    ax.imshow(ps.visualization.show_3D(im), interpolation='none')
    ax.set_title("BLOBS")
    plt.show()

    imageio.volsave("Blobs.tif", np.array(im.astype("uint8")))
    return im


def CYLINDERS(shape, radius=2, ncylinders=10, porosity=0.5, cyl_length=None, phi_max=90, theta_max=90, maxiter=10, invert=False):   #returns im
    # By controlling the number of cyliders
    # radius: radius of cylinders
    # ncylinders: number of cyl
    # cyl_length: cylinder length
    # phi_max: maximum angle of rotation around z-axis (default=90)
    # theta_max: maximum angle of rotation around y-axis (default=90)
    # maxiter: maximum number of iterations to match the desired porosity
    # invert: if True, inverts the image (pores become solid and vice versa)

    if cyl_length is not None:
        im = ps.generators.cylinders(shape=shape, r = radius, porosity = porosity, length = cyl_length, phi_max=phi_max, theta_max=theta_max, maxiter=maxiter)    # By controlling the porosity

    else:
        im = ps.generators.cylinders(shape=shape, r = radius, ncylinders = ncylinders, length = cyl_length, phi_max=phi_max, theta_max=theta_max)

    if invert:
        im = ~im
    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    ax.imshow(ps.visualization.show_3D(im), interpolation='none')
    ax.set_title("CYLINDERS")
    plt.show()

    imageio.volsave("Cylinders.tif", np.array(im.astype("uint8")))
    return im


def OVERLAPPING_SPHERES(shape, radius, porosity, maxiter=10, invert=False):  #returns im
    # r: controls size of spheres, maxiter: number of of iterations used to match the desired porosity default=10
    # invert: if True, inverts the image (pores become solid and vice versa)

    im = ps.generators.overlapping_spheres(shape=shape, r = radius, porosity=porosity, maxiter=maxiter)
    if invert:
        im = ~im
    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    ax.imshow(ps.visualization.show_3D(im), interpolation='none')
    ax.set_title("OVERLAPPING SPHERES")
    ax.axis(False)
    print('porosity = ', ps.metrics.porosity(im))
    plt.show()

    imageio.volsave("Overlapping_Spheres.tif", np.array(im.astype("uint8")))
    return im


def POLYDISPERSE_SPHERES(shape, porosity=0.5, r_min=4, nbins=3, invert=False):  #returns im
    # dist: statistical distribution of spheres diameters
    # nbins: number of discrete sphere sizes (default=5), r_min: smallest sphere to generate (default=5)
    # invert: if True, inverts the image (pores become solid and vice versa)

    dist = spst.norm(loc=10, scale=5)
    im = ps.generators.polydisperse_spheres(shape=shape, porosity = porosity, nbins=nbins, r_min=r_min, dist=dist)
    if invert:
        im = ~im
    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    ax.imshow(ps.visualization.show_3D(im), interpolation='none')
    ax.set_title("POLYDISPERSE SPHERES")
    ax.axis(False)
    plt.show()
    
    imageio.volsave("Polydisperse_Spheres.tif", np.array(im.astype("uint8")))
    return im


def PSEUDO_GRAVITY_PACKING(shape, radius=3, maxiter=50, clearance=0, pk_axis=0, invert=False):  #returns im
    # r: radius of spheres
    # clearance: <0 for overlaping between spheres
    # maxiter: maximum number of spheres to add
    # pk_axis: direction of gravity (default axis=0 -> x=0)
    # invert: if True, inverts the image (pores become solid and vice versa)

    im = ~ps.generators.pseudo_gravity_packing(shape=shape, r=radius, maxiter=maxiter, clearance=clearance, axis=pk_axis)
    if invert:
        im = ~im
    fig, ax = plt.subplots(1, figsize=[6, 6])
    ax.imshow(ps.visualization.show_3D(im), interpolation='none', origin='lower')
    ax.set_title("PSEUDO GRAVITY PACKING")
    ax.axis(False)
    plt.show()

    imageio.volsave("Pseudo_Gravity_Packing.tif", np.array(im.astype("uint8")))
    return im


def VORONOI_EDGES(shape, thickness, ncells, flat_faces = True, invert=False):  #returns im
    # ncells: number of cells
    # flat_faces = false : for edges to be true, r: thickness of edge (default=1)
    # invert: if True, inverts the image (pores become solid and vice versa)

    im = ps.generators.voronoi_edges(shape=shape, r=thickness, ncells=ncells, flat_faces=flat_faces)
    if invert:
        im = ~im
    fig, ax = plt.subplots(figsize=[6, 6])
    ax.imshow(ps.visualization.show_3D(~im), interpolation='none')          # remove ~ to make edges solid
    ax.set_title("VORONOI EDGES")
    ax.axis(False)
    plt.show()

    imageio.volsave("Voronoi_Edges.tif", np.array(im.astype("uint8")))
    return im

