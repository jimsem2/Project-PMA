import matplotlib.pyplot as plt
import numpy as np
import porespy as ps
from porespy.filters import fftmorphology
from skimage.morphology import ball
from edt import edt
from sympy import im

ps.visualization.set_mpl_style()
np.random.seed(10)

# YELLOW = TRUE = 1 = VOID/PORE // DARK BLUE = FALSE = 0 = SOLID

def BOND_NUMBER(im, g=9.81, sigma=0.01, delta_rho=1000, voxel_size=0.001, source='lt', method='median'): # returns bn
    # im: binary image from Generators_3D
    # g: gravitational acceleration, default 9.81 m/s^2
    # sigma: surface tension, default 0.01 N/m
    # delta_rho: density difference, default 1000 kg/m^3
    # voxel_size: used for scaling, 1 (default)
    # source: how the characteristic size is calculated, default 'lt', options: 'dt' for distance transform, 'lt' for lineal path distribution
    # method: how the values from 'source'are averaged, default 'median', options: 'mean', 'median', 'min', 'max'

    bn = ps.metrics.bond_number(im=im, g=g, sigma=sigma, delta_rho=delta_rho, voxel_size=voxel_size, source=source, method=method)
    print ('Bond number is', bn)

    return bn


def CHORD_COUNTS(im, spacing=1, trim_edges=True, apply_chords3D=True, axis=0, bins=25): # returns cc
    # works both with apply_chords and apply_chords_3D
    # Applies chords and counts their lengths
    # spacing: space between chords, trim_edges: to remove edges that touch borders
    # apply_chords3D: if True applies apply_chords_3D, if False applies apply_chords
    # axis: the axis along which chords are drawn, only used when apply_chords3D is False, options: 0 for x axis, 1 for y axis, 2 for z axis
    # bins: number of classes for the histogram, 25 (default)

    if apply_chords3D:
        chords = ps.filters.apply_chords_3D(im=im, spacing=spacing, trim_edges=trim_edges)
        cc = ps.metrics.chord_counts(im=chords)

        fig, ax = plt.subplots(1, figsize=[6, 6])
        ax.hist(cc, bins=bins)
        ax.set_xlabel('chord length')
        ax.set_ylabel('frequency')
    else:
        chords = ps.filters.apply_chords(im=im, spacing=spacing, axis=axis, trim_edges=trim_edges, label=True)
        cc = ps.metrics.chord_counts(im=chords)

        fig, ax = plt.subplots(1, figsize=[6, 6])
        ax.hist(cc, bins=bins)
        ax.set_xlabel('chord length along selected axis')
        ax.set_ylabel('frequency')

    plt.show()
    return cc


def CHORD_LENGTH_DISTRIBUTION(im, bins=25, log=False, voxel_size=0.001, normalization='number', apply_chords3D=True, axis=0):   # returns cld_data
    # im: binary image from Generators_3D
    # bins: number of classes for the histogram, 25 (default)
    # log: logarithmic scale, if True then 'data.L' must be 'data.LogL' in the diagrams
    # voxel_size: used for scaling, 1 (default)
    # normalization: how to normalize bin heights, options: 'count' or 'number' (default), 'length'
    # axis: the axis along which chords are drawn, only used when apply_chords3D is False, options: 0 for x axis, 1 for y axis, 2 for z axis

    if apply_chords3D:
        im_chords = ps.filters.apply_chords_3D(im=im)
    else:
        im_chords = ps.filters.apply_chords(im=im, axis=axis)
    cld_data = ps.metrics.chord_length_distribution(im=im_chords, bins=bins, log=log, voxel_size=voxel_size, normalization=normalization)

    fig, ax = plt.subplots(1, 3, figsize=[12, 4])
    ax[0].plot(cld_data.L, cld_data.pdf)
    ax[0].set_title("Probability Density Function")
    ax[1].plot(cld_data.L, cld_data.cdf)
    ax[1].set_title("Cumulative Density Function")
    ax[2].bar(cld_data.L, cld_data.cdf, cld_data.bin_widths, edgecolor='k')
    ax[2].set_title('Bar Plot')

    plt.show()
    return cld_data


def FIND_H(im, srange=[0.01, 0.93], axis_of_intrusion=0, voxel_size=1, g=0, s=0.5): # returns h
    # im: binary image from Generators_3D
    # srange: saturation range to apply find_h
    # axis_of_intrusion: the axis along which the intrusion happens, options: 0 for x axis, 1 for y axis, 2 for z axis
    # voxel_size: used for scaling, 1 (default)
    # g: gravitational acceleration, 0 (default)
    # s: global saturation at which to find h, 0.5 (default)

    inlets = np.zeros_like(im)
    if axis_of_intrusion == 0:
        inlets[0, :, :] = True
    elif axis_of_intrusion == 1:
        inlets[:, 0, :] = True
    elif axis_of_intrusion == 2:
        inlets[:, :, 0] = True
    pc = ps.filters.capillary_transform(im=im, voxel_size=voxel_size, g=g)
    inv = ps.simulations.drainage(im=im, inlets=inlets, pc=pc)
    s_prof = ps.metrics.satn_profile(satn=inv.im_snwp, s=s)

    fig, ax = plt.subplots(1, figsize=[6, 6])
    ax.plot(s_prof.position, s_prof.saturation, 'b-o')
    ax.set_xlabel("distance from injection face")
    ax.set_ylabel("non-wetting phase saturation")
   
    h = ps.metrics.find_h(s_prof.saturation, s_prof.position, srange=srange)
    print(h)

    plt.show()
    return h


def FIND_POROSITY_THRESHOLD(im, axis=0, conn="max"):   # returns threshold
    # im: binary image from Generators_3D
    # axis: the axis along which the intrusion happens, options: 0 for x axis, 1 for y axis, 2 for z axis
    # conn:  Options: max (default, when a corner is considered connection between voxels), min (when only face is a connection)

    threshold = ps.metrics.find_porosity_threshold(im=im, axis=axis, conn=conn)
    print(threshold)

    return threshold


def IS_PERCOLATING(im, conn="max"):
    # im: binary image from Generators_3D
    # conn:  Options: max (default, when a corner is considered connection between voxels), min (when only face is a connection)

    perc0 = ps.metrics.is_percolating(im=im, conn=conn, axis=0)
    perc1 = ps.metrics.is_percolating(im=im, conn=conn, axis=1)
    perc2 = ps.metrics.is_percolating(im=im, conn=conn, axis=2)
    print('is the image percolating in X axis?', perc0)
    print('is the image percolating in Y axis?', perc1)
    print('is the image percolating in Z axis?', perc2)


def LINEAL_PATH_DISTRIBUTION(im, bins=25, log=False, voxel_size=1): # returns lpd_data
    # im: binary image from Generators_3D
    # bins: number of classes for the histogram, 25 (default)
    # log: logarithmic scale, if True then 'data.L' must be 'data.LogL' in the diagrams
    # voxel_size: used for scaling, 1 (default)

    im_lin = ps.filters.distance_transform_lin(im=im, axis=0, mode='forward')
    lpd_data = ps.metrics.lineal_path_distribution(im_lin, bins=bins, log=log, voxel_size=voxel_size)

    fig, ax = plt.subplots(1, 3, figsize=[10, 4])
    ax[0].plot(lpd_data.L, lpd_data.pdf)
    ax[0].set_title("Probability Density Function")
    ax[1].plot(lpd_data.L, lpd_data.cdf)
    ax[1].set_title("Cumulative Density Function")
    ax[2].bar(lpd_data.L, lpd_data.cdf, lpd_data.bin_widths, edgecolor='k')
    ax[2].set_title('Bar Plot')

    plt.show()
    return lpd_data


def MESH_SURFACE_AREA(im, pore_label=None): # returns surface_area
    # im: binary image from Generators_3D
    # pore_label: if provided, calculates area of that specific pore region

    mesh = ps.tools.mesh_region(region=im)
    surface_area = ps.metrics.mesh_surface_area(mesh=mesh)
    print('surface area of void space is', str(surface_area))

    if pore_label is not None:
        snow = ps.filters.snow_partitioning(im=im)
        regions = snow.regions / im
        pore_region = regions == pore_label
        mesh = ps.tools.mesh_region(region=pore_region)
        area = ps.metrics.mesh_surface_area(mesh=mesh)
        print('surface area of the meshed pore region is', str(area))
        return area

    return surface_area


def MESH_VOLUME(im, pore_label=None):   # returns volume
    # im: binary image from Generators_3D
    # pore_label: if provided, calculates volume of that specific pore region

    volume = ps.metrics.mesh_volume(region=im)
    print('Total volume is =', volume)

    if pore_label is not None:
        snow = ps.filters.snow_partitioning(im=im)
        regions = snow.regions / im
        pore_region = regions == pore_label
        volume = ps.metrics.mesh_volume(region=pore_region)
        print('Selected pore volume is = ', volume)

    return volume


def PC_MAP_TO_PC_CURVE(im, sigma=0.01, theta=180, voxel_size=1, steps=25): # returns pc_curve
    # im: binary image from Generators_3D
    # sigma: surface tension of the fluid-fluid system of interest
    # theta: contact angle measured through the invading phase in degrees
    # voxel_size: used for scaling, 1 (default)

    pc = ps.filters.capillary_transform(im=im, sigma=sigma, theta=theta, voxel_size=voxel_size)
    drn = ps.simulations.drainage(im=im, pc=pc, steps=steps)
    pc_curve = ps.metrics.pc_map_to_pc_curve(im=im, pc=drn.im_pc)

    print(pc_curve)

    fig, ax = plt.subplots(1, 3, figsize=[15, 5])
    ax[0].plot(pc_curve.pc, pc_curve.snwp, 'b-o')
    ax[0].set_title('Plot diagram')
    ax[0].set_xlabel('Capillary Pressure [Pa]')
    ax[0].set_ylabel('Non-wetting Phase Saturation')
    ax[1].step(pc_curve.pc, pc_curve.snwp, 'b-o', where='post')
    ax[1].set_title('Step diagram')
    ax[1].set_xlabel('Capillary Pressure [Pa]')
    ax[1].set_ylabel('Non-wetting Phase Saturation')
    ax[2].semilogx(pc_curve.pc, pc_curve.snwp, 'r-o')
    ax[2].set_title('Logarithmic CP')
    ax[2].set_xlabel('Capillary Pressure [Pa]')
    ax[2].set_ylabel('Non-wetting Phase Saturation')

    plt.show()
    return pc_curve


def PERCOLATING_POROSITY(im, conn="max"):   # returns eps0, eps1, eps2
    # im: binary image from Generators_3D
    # conn:  Options: max (default, when a corner is considered connection between voxels), min (when only face is a connection)

    eps0 = ps.metrics.percolating_porosity(im=im, conn=conn, axis=0)
    eps1 = ps.metrics.percolating_porosity(im=im, conn=conn, axis=1)
    eps2 = ps.metrics.percolating_porosity(im=im, conn=conn, axis=2)
    print('percolating porosity in x-axis is', eps0)
    print('percolating porosity in y-axis is', eps1)
    print('percolating porosity in z-axis is', eps2)

    return eps0, eps1, eps2


def PORE_SIZE_DISTRIBUTION(im, bins=25, voxel_size=1, log=False):   # returns psd_data
    # im: binary image from Generators_3D
    # bins: number of classes for the histogram, 25 (default)
    # voxel_size: used for scaling, 1 (default)
    # log: if False the histogram binning is based on linear(pore radius)

    im_poro = ps.filters.porosimetry(im=im)
    psd_data = ps.metrics.pore_size_distribution(im=im_poro, bins=bins, voxel_size=voxel_size, log=log)

    fig, ax = plt.subplots(1, 3, figsize=[15, 5])
    ax[0].plot(psd_data.bin_centers, psd_data.pdf)
    ax[0].set_title("Probability Density Function")
    ax[0].set_xlabel('pore radius in voxels')
    ax[0].set_ylabel('frequency')
    ax[1].plot(psd_data.bin_centers, psd_data.cdf)
    ax[1].set_title("Cumulative Density Function")
    ax[1].set_xlabel('pore radius in voxels')
    ax[1].set_ylabel('cumulative frequency')
    ax[2].bar(psd_data.bin_centers, psd_data.cdf, psd_data.bin_widths, edgecolor='k')
    ax[2].set_title('Bar Plot')
    ax[2].set_xlabel('pore radius in voxels')
    ax[2].set_ylabel('cumulative frequency')

    plt.show()
    return psd_data


def POROSITY(im):   # returns porosity
    # im: binary image from Generators_3D

    porosity = ps.metrics.porosity(im=im)
    print('Porosity:', porosity)

    return porosity


def POROSITY_PROFILE(im, voxel_size=1):   # returns prfx, prfy, prfz
    # im: binary image from Generators_3D
    # voxel_size: microns/voxel, default 1

    prfx = ps.metrics.porosity_profile(im=im, axis=0)
    prfy = ps.metrics.porosity_profile(im=im, axis=1)
    prfz = ps.metrics.porosity_profile(im=im, axis=2)

    fig, ax = plt.subplots(1, figsize=[6, 6])
    ax.plot(prfx.position * voxel_size, prfx.porosity, 'b-', label='yz-plane', alpha=0.5)
    ax.plot(prfy.position * voxel_size, prfy.porosity, 'r-', label='xz-plane', alpha=0.5)
    ax.plot(prfz.position * voxel_size, prfz.porosity, 'g-', label='xy-plane', alpha=0.5)
    ax.plot([0, 1], [ps.metrics.porosity(im=im), ps.metrics.porosity(im=im)], 'r--')
    ax.set_ylabel('Porosity of slice')
    ax.set_xlabel('Position of slice along given axis')
    ax.legend()

    fig, ax = plt.subplots(1, figsize=[6, 6])
    ax.plot(prfx.position / prfx.position.max(), prfx.porosity, 'b-', label='yz-plane', alpha=0.5)
    ax.plot(prfy.position / prfy.position.max(), prfy.porosity, 'r-', label='xz-plane', alpha=0.5)
    ax.plot(prfz.position / prfz.position.max(), prfz.porosity, 'g-', label='xy-plane', alpha=0.5)
    ax.plot([0, 1], [ps.metrics.porosity(im=im), ps.metrics.porosity(im=im)], 'r--')
    ax.set_ylabel('Porosity of slice')
    ax.set_xlabel('Fractional distance along given axis')
    ax.legend()

    plt.show()
    return prfx, prfy, prfz


def RADIAL_DENSITY_DISTRIBUTION(im, bins=25, log=False, voxel_size=1):  # returns rdd_data
    # im: binary image from Generators_3D
    # bins: number of classes for the histogram, 25 (default)
    # log: if True the histogram binning is based on log(size)
    # voxel_size: used for scaling, 1 (default)

    dt = edt(im)
    rdd_data = ps.metrics.radial_density_distribution(dt=dt, bins=bins, log=log, voxel_size=voxel_size)

    fig, ax = plt.subplots(1, 3, figsize=[12, 4])
    ax[0].plot(rdd_data.bin_centers, rdd_data.pdf)
    ax[0].set_title("Probability Density Function")
    ax[1].plot(rdd_data.bin_centers, rdd_data.cdf)
    ax[1].set_title("Cumulative Density Function")
    ax[2].bar(rdd_data.bin_centers, rdd_data.cdf, rdd_data.bin_widths, edgecolor='k')
    ax[2].set_title('Bar Plot')

    plt.show()
    return rdd_data


def REGION_SURFACE_AREAS(im, voxel_size=1, strel_size=2, slice=0):   # returns surface_areas
    # im: binary image from Generators_3D
    # voxel_size: used for scaling, 1 (default)
    # strel: structuring element used to blur the region, bigger size means region will become increasingly blurred and inaccurate
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z

    snow = ps.filters.snow_partitioning(im=im)
    regions = snow.regions
    surface_areas = ps.metrics.region_surface_areas(regions=regions, voxel_size=voxel_size, strel=ball(strel_size))

    # Labeling each region by its surface
    regions_sur = np.copy(regions)
    for label, area in enumerate(surface_areas, start=1):
        mask = regions == label
        regions_sur[mask] = area

    fig, ax = plt.subplots(1, 2, figsize=[12, 5])
    ind = int(im.shape[slice]/2)
    if slice == 0:
        im_plot = ax[0].imshow((regions_sur)[ind, :, :], interpolation='none')
        ax[0].set_title('YZ slice - Surface area of regions')
    elif slice == 1:
        im_plot = ax[0].imshow((regions_sur)[:, ind, :], interpolation='none')
        ax[0].set_title('XZ slice - Surface area of regions')
    else:
        im_plot = ax[0].imshow((regions_sur)[:, :, ind], interpolation='none')
        ax[0].set_title('XY slice - Surface area of regions')
    fig.colorbar(im_plot, ax=ax[0], shrink=0.8)
    ax[0].axis(False)

    ax[1].hist(surface_areas, edgecolor='k')
    ax[1].set_title("Probability Density Function")

    plt.show()
    return surface_areas


def REGION_VOLUMES(im, method='voxel', slice=0):    # returns r_volumes
    # im: binary image from Generators_3D
    # method: how the volume of each region is calculated, options: 'marching_cubes' (DEF) & 'voxel'
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z

    snow = ps.filters.snow_partitioning(im=im)
    regions = snow.regions
    r_volumes = ps.metrics.region_volumes(regions=regions, method=method)

    regions_vol = np.copy(regions)
    for label, volume in enumerate(r_volumes, start=1):
        mask = regions == label
        regions_vol[mask] = volume

    fig, ax = plt.subplots(1, 2, figsize=[12, 5])
    ind = int(im.shape[slice]/2)
    if slice == 0:
        rgs = ax[0].imshow((regions_vol)[ind, :, :], interpolation='none')
        ax[0].set_title('YZ slice - Volume of regions')
    elif slice == 1:
        rgs = ax[0].imshow((regions_vol)[:, ind, :], interpolation='none')
        ax[0].set_title('XZ slice - Volume of regions')
    else:
        rgs = ax[0].imshow((regions_vol)[:, :, ind], interpolation='none')
        ax[0].set_title('XY slice - Volume of regions')
    fig.colorbar(rgs, ax=ax[0], shrink=0.8)
    ax[0].axis(False)

    ax[1].hist(r_volumes, edgecolor='k')
    ax[1].set_title("Probability Density Function")

    plt.show()
    return r_volumes


def REV_POROSITY(im, n=1000):  # returns rev_porosity
    # im: binary image from Generators_3D
    # n: number of randomly located sized boxes, default 1000

    rev_porosity = ps.metrics.rev_porosity(im=im)

    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    ax.plot(rev_porosity.volume, rev_porosity.porosity, "r.")
    ax.set_xlabel("volume")
    ax.set_ylabel("porosity")

    plt.show()
    return rev_porosity


def SATN_PROFILE(im, span=2, mode='tile', axis=0, sigma=0.01, theta=180, g=0):  # returns sp_list
    # im: binary image from Generators_3D
    # span: width of slice over which the saturation is computed, default 2 voxels
    # mode: How averaging window moves, options: by sliding ('slide') or by tiling ('tile')
    # axis: the axis along which the intrusion happens, options: 0 for x axis, 1 for y axis, 2 for z axis
    # sigma: surface tension of the fluid-fluid system of interest
    # theta: contact angle measured through the invading phase in degrees
    # g: gravitational acceleration, 0 (default)

    inlets = np.zeros_like(im)
    if axis == 0:
        inlets[0, :, :] = True
    elif axis == 1:
        inlets[:, 0, :] = True
    else:
        inlets[:, :, 0] = True
    
    pc = ps.filters.capillary_transform(im=im, sigma=sigma, theta=theta, g=g)
    inv = ps.simulations.drainage(im=im, pc=pc, inlets=inlets)

    fig, ax = plt.subplots(1, figsize=[6, 6])
    s_values = np.arange(0.1, 1.0, 0.2)
    sp_list = []
    for s_value in s_values:
        sp = ps.metrics.satn_profile(satn=inv.im_snwp, s=s_value, span=span, mode=mode, axis=axis)
        sp_list.append(sp)
        ax.plot(sp.position, sp.saturation, '-o', label=f's={s_value:.1f}')
    ax.set_xlabel("distance from injection face")
    ax.set_ylabel("non-wetting phase saturation")
    ax.legend()

    plt.show()
    return sp_list


