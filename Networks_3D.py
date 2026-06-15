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

    # Applying snow partitioning
    snow_output = ps.networks.snow2(im, voxel_size=1)
    # Creating network
    pn = op.io.network_from_porespy(snow_output.network)

    # Assigning values to network
    pn['pore.values'] = np.random.rand(pn.Np)
    # Assigning the values to the image
    fig, ax = plt.subplots()
    reg = ps.networks.map_to_regions(regions=snow_output.regions.T, values=pn['pore.values'])
    plt.imshow((reg)[:, :, ind])
    plt.show()

