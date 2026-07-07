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
    # shape: 3D shape of the image (x, y, z)
    # porosity: percentage of void space. When: porosity=None -> return: greyscale image
    # blobiness: integer/list, higher number -> more blobs
    # invert: if True, inverts the image (pores become solid and vice versa)

    np.random.seed(10)
    im = ps.generators.blobs(shape=shape, porosity=porosity, blobiness=blobiness)
    if invert:
        im = ~im
    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    fig.canvas.manager.set_window_title("BLOBS")
    ax.imshow(ps.visualization.show_3D(im), interpolation='none')
    ax.set_title("BLOBS")
    plt.show(block=False)

    imageio.volsave("Blobs.tif", np.transpose(np.array(im.astype("uint8") * 255), (2, 1, 0)))
    return im


def CYLINDERS(shape, radius=4, ncylinders=25, porosity=0.5, cyl_length=None, phi_max=90, theta_max=90, maxiter=10, invert=False):   #returns im
    # shape: 3D shape of the image (x, y, z)
    # radius: radius of cylinders
    # ncylinders: number of cylinders to generate (default=25)
    # porosity: percentage of void space
    # cyl_length: cylinder length. When: cyl_length=None -> cylinders span the entire domain. When: cyl_length=scalar -> number of cylinders to generate is determined by porosity.
    # phi_max: maximum angle of rotation around z-axis (default=90)
    # theta_max: maximum angle of rotation around y-axis (default=90)
    # maxiter: maximum number of iterations to match the desired porosity
    # invert: if True, inverts the image (pores become solid and vice versa)

    np.random.seed(10)
    if cyl_length is not None:
        im = ps.generators.cylinders(shape=shape, r = radius, porosity = porosity, length = cyl_length, phi_max=phi_max, theta_max=theta_max, maxiter=maxiter)    # By controlling the porosity

    else:
        im = ps.generators.cylinders(shape=shape, r = radius, ncylinders = ncylinders, length = cyl_length, phi_max=phi_max, theta_max=theta_max)

    if invert:
        im = ~im
    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    fig.canvas.manager.set_window_title("CYLINDERS")
    ax.imshow(ps.visualization.show_3D(im), interpolation='none')
    ax.set_title("CYLINDERS")
    plt.show(block=False)

    imageio.volsave("Cylinders.tif", np.transpose(np.array(im.astype("uint8") * 255), (2, 1, 0)))
    return im


def OVERLAPPING_SPHERES(shape, radius=3, porosity=0.5, maxiter=10, invert=False):  #returns im
    # shape: 3D shape of the image (x, y, z)
    # porosity: percentage of void space
    # r: controls radius of spheres
    # maxiter: number of of iterations used to match the desired porosity default=10
    # invert: if True, inverts the image (pores become solid and vice versa)

    np.random.seed(10)
    im = ps.generators.overlapping_spheres(shape=shape, r = radius, porosity=porosity, maxiter=maxiter)
    if invert:
        im = ~im
    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    fig.canvas.manager.set_window_title("OVERLAPPING SPHERES")
    ax.imshow(ps.visualization.show_3D(im), interpolation='none')
    ax.set_title("OVERLAPPING SPHERES")
    ax.axis(False)
    plt.show(block=False)

    imageio.volsave("Overlapping_Spheres.tif", np.transpose(np.array(im.astype("uint8") * 255), (2, 1, 0)))
    return im


def POLYDISPERSE_SPHERES(shape, porosity=0.5, r_min=3, nbins=5, loc=10, scale=5, invert=False):  #returns im
    # shape: 3D shape of the image (x, y, z)
    # porosity: percentage of void space
    # r_min: smallest sphere to generate (default=3)
    # nbins: number of discrete sphere sizes (default=5)
    # loc: location parameter for the normal distribution (default=10). The bigger the loc, the bigger the spheres.
    # scale: scale parameter for the normal distribution (default=5). The bigger the scale, the more variation in sphere sizes.
    # invert: if True, inverts the image (pores become solid and vice versa)

    np.random.seed(10)
    dist = spst.norm(loc=loc, scale=scale)               # dist: statistical distribution of spheres diameters
    im = ps.generators.polydisperse_spheres(shape=shape, porosity = porosity, nbins=nbins, r_min=r_min, dist=dist)
    if invert:
        im = ~im
    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    fig.canvas.manager.set_window_title("POLYDISPERSE SPHERES")
    ax.imshow(ps.visualization.show_3D(im), interpolation='none')
    ax.set_title("POLYDISPERSE SPHERES")
    ax.axis(False)
    plt.show(block=False)
    
    imageio.volsave("Polydisperse_Spheres.tif", np.transpose(np.array(im.astype("uint8") * 255), (2, 1, 0)))
    return im


def PSEUDO_ELECTROSTATIC_PACKING(shape, r=3, clearance=0, protrusion=0, maxiter=500, phi=0.3, compactness=0.5, invert=False):  #returns im
    # shape: 3D shape of the image (x, y, z)
    # r: radius of spheres
    # clearance: distance between spheres (negative for overlap)
    # maxiter: maximum number of spheres to add
    # protrusion: amount by which spheres extend beyond the domain
    # phi: solid volume fraction of the spheres. It overrides the maxiter parameter if provided when adds fewer spheres.
    # compactness: controls the compactness of the packing. <1 for relaxed packing.
    # invert: if True, inverts the image (pores become solid and vice versa)

    np.random.seed(10)
    im_sites = ps.generators.borders(shape=shape, mode="faces")
    kwargs = dict(r=r, maxiter=maxiter, clearance=clearance, protrusion=protrusion, compactness=compactness)
    if phi is not None:
        kwargs['phi'] = phi
    im = ps.generators.pseudo_electrostatic_packing(im=im_sites, **kwargs).astype(bool)
    im = im & ~im_sites  # remove face borders; keep only spheres
    im = ~im  # convert to porespy convention: True = void
    if invert:
        im = ~im
    fig, ax = plt.subplots(1, figsize=[6, 6])
    fig.canvas.manager.set_window_title("PSEUDO ELECTROSTATIC PACKING")
    ax.imshow(ps.visualization.show_3D(im), interpolation='none', origin='lower')
    ax.set_title("PSEUDO ELECTROSTATIC PACKING")
    ax.axis(False)
    plt.show(block=False)

    imageio.volsave("Pseudo_Electrostatic_Packing.tif", np.transpose(np.array(im.astype("uint8") * 255), (2, 1, 0)))
    return im


def PSEUDO_GRAVITY_PACKING(shape, r=5, maxiter=200, clearance=0, pk_axis=0, invert=False):  #returns im
    # shape: 3D shape of the image (x, y, z)
    # r: radius of spheres
    # clearance: distance between spheres (negative for overlap)
    # maxiter: maximum number of spheres to add
    # pk_axis: direction of gravity (default axis=0 -> x=0)
    # invert: if True, inverts the image (pores become solid and vice versa)

    np.random.seed(10)
    im = ps.generators.pseudo_gravity_packing(shape=shape, r=r, maxiter=maxiter, clearance=clearance, axis=pk_axis).astype(bool)
    im = ~im  # convert to porespy convention: True = void
    if invert:
        im = ~im
    fig, ax = plt.subplots(1, figsize=[6, 6])
    fig.canvas.manager.set_window_title("PSEUDO GRAVITY PACKING")
    ax.imshow(ps.visualization.show_3D(im), interpolation='none')
    ax.set_title("PSEUDO GRAVITY PACKING")
    ax.axis(False)
    plt.show(block=False)

    imageio.volsave("Pseudo_Gravity_Packing.tif", np.transpose(np.array(im.astype("uint8") * 255), (2, 1, 0)))
    return im


def VORONOI_EDGES(shape, thickness=3, ncells=50, flat_faces = True, invert=False):  #returns im
    # shape: 3D shape of the image (x, y, z)
    # thickness: thickness of the edges (default=3)
    # ncells: number of cells
    # flat_faces = false : for edges to be true
    # invert: if True, inverts the image (pores become solid and vice versa)

    np.random.seed(10)
    im = ps.generators.voronoi_edges(shape=shape, r=thickness, ncells=ncells, flat_faces=flat_faces)
    if invert:
        im = ~im
    fig, ax = plt.subplots(figsize=[6, 6])
    fig.canvas.manager.set_window_title("VORONOI EDGES")
    ax.imshow(ps.visualization.show_3D(im), interpolation='none')          # remove ~ to make edges solid
    ax.set_title("VORONOI EDGES")
    ax.axis(False)
    plt.show(block=False)

    imageio.volsave("Voronoi_Edges.tif", np.transpose(np.array(im.astype("uint8") * 255), (2, 1, 0)))
    return im


