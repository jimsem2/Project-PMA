import matplotlib.pyplot as plt
import numpy as np
import porespy as ps
from porespy.filters import fftmorphology
from skimage.morphology import ball
from edt import edt

ps.visualization.set_mpl_style()
np.random.seed(10)

# YELLOW = TRUE = 1 = VOID/PORE // DARK BLUE = FALSE = 0 = SOLID


def CHORD_COUNTS(im):
    # im: binary image from Generators_3D
    # Applies chords and counts their lengths

    im_chords = ps.filters.apply_chords(im)
    chord_length = ps.metrics.chord_counts(im=im_chords)

    length = im.shape[0]
    fig, ax = plt.subplots(1, 3, figsize=[12, 4])
    ind = int(length/2)
    ax[0].imshow((im_chords)[:, :, ind], interpolation='none')
    ax[0].axis(False)
    ax[1].hist(chord_length, bins=100)
    ax[1].set_xlabel('chord length along x axis')
    ax[1].set_ylabel('frequency')
    ax[2].plot(np.bincount(chord_length))
    ax[2].set_xlabel('chord length')
    ax[2].set_ylabel('the number of chords of each length')

    plt.show()
    return chord_length


def CHORD_LENGTH_DISTRIBUTION(im, bins=30, log=False, voxel_size=0.001, normalization='length'):
    # im: binary image from Generators_3D
    # bins: number of classes for the histogram, 10 (default)
    # log: logarithmic scale, if True then 'data.L' must be 'data.LogL' in the diagrams
    # voxel_size: used for scaling, 1 (default)
    # normalization: how to normalize bin heights, options: 'count' or 'number' (default), 'length'

    im_chords = ps.filters.apply_chords_3D(im)
    data = ps.metrics.chord_length_distribution(im=im_chords, bins=bins, log=log, voxel_size=voxel_size, normalization=normalization)

    fig, ax = plt.subplots(1, 3, figsize=[12, 4])
    ax[0].plot(data.L, data.pdf)
    ax[0].set_title("Probability Density Function")
    ax[1].plot(data.L, data.cdf)
    ax[1].set_title("Cumulative Density Function")
    ax[2].bar(data.L, data.cdf, data.bin_widths, edgecolor='k')
    ax[2].set_title('Bar Plot')

    plt.show()
    return data


def FIND_H(im, srange=[0.01, 0.93]):
    # im: binary image from Generators_3D
    # srange: saturation range to apply find_h

    inlets = np.zeros_like(im)
    inlets[0, :, :] = True
    inv = ps.simulations.drainage(im=im, inlets=inlets, voxel_size=1, g=0, bins=25)
    s = ps.metrics.satn_profile(satn=inv.im_satn, s=0.6)

    fig, ax = plt.subplots(1, figsize=[6, 6])
    ax.plot(s.position, s.saturation, 'b-o')
    ax.set_xlabel("distance from injection face")
    ax.set_ylabel("non-wetting phase saturation")

    h = ps.metrics.find_h(s.saturation, s.position, srange=srange)
    print(h)

    plt.show()
    return h


def LINEAL_PATH_DISTRIBUTION(im, bins=25, log=False, voxel_size=0.01):
    # im: binary image from Generators_3D
    # bins: number of classes for the histogram, 10 (default)
    # log: logarithmic scale, if True then 'data.L' must be 'data.LogL' in the diagrams
    # voxel_size: used for scaling, 1 (default)

    im_lin = ps.filters.distance_transform_lin(im=im, axis=0, mode='forward')
    data = ps.metrics.lineal_path_distribution(im_lin, bins=bins, log=log, voxel_size=voxel_size)

    fig, ax = plt.subplots(1, 3, figsize=[10, 4])
    ax[0].plot(data.L, data.pdf)
    ax[0].set_title("Probability Density Function")
    ax[1].plot(data.L, data.cdf)
    ax[1].set_title("Cumulative Density Function")
    ax[2].bar(data.L, data.cdf, data.bin_widths, edgecolor='k')
    ax[2].set_title('Bar Plot')

    plt.show()
    return data


def MESH_SURFACE_AREA(im, pore_label=None):
    # im: binary image from Generators_3D
    # pore_label: if provided, calculates area of that specific pore region

    mesh = ps.tools.mesh_region(region=im)
    surface_area = ps.metrics.mesh_surface_area(mesh=mesh)
    print('surface area of void space is', str(surface_area))

    if pore_label is not None:
        snow = ps.filters.snow_partitioning(im)
        regions = snow.regions / im
        pore_region = regions == pore_label
        mesh = ps.tools.mesh_region(region=pore_region)
        ps.visualization.show_mesh(mesh)
        area = ps.metrics.mesh_surface_area(mesh=mesh)
        print('surface area of the meshed pore region ' + str(area))
        return area

    return surface_area


def MESH_VOLUME(im, pore_label=None):
    # im: binary image from Generators_3D
    # pore_label: if provided, calculates volume of that specific pore region

    volume = ps.metrics.mesh_volume(region=im)
    print('Total volume is =', volume)

    if pore_label is not None:
        snow = ps.filters.snow_partitioning(im)
        regions = snow.regions / im
        pore_region = regions == pore_label
        volume = ps.metrics.mesh_volume(region=pore_region)
        print('Selected pore volume is = ', volume)

    return volume


def PC_CURVE(im, sigma=0.072, theta=180, voxel_size=1e-5):
    # im: binary image from Generators_3D
    # sigma: surface tension of the fluid-fluid system of interest
    # theta: contact angle measured through the invading phase in degrees

    drainage = ps.filters.porosimetry(im)
    pc = -2*sigma*np.cos(np.deg2rad(theta))/(drainage*voxel_size)
    data = ps.metrics.pc_curve(im=im, sizes=drainage, pc=pc, voxel_size=voxel_size)

    print(data)

    fig, ax = plt.subplots(1, 3, figsize=[18, 6])
    ax[0].plot(data.pc, data.snwp, 'b-o')
    ax[0].set_title('Plot diagram')
    ax[0].set_xlabel('Capillary Pressure [Pa]')
    ax[0].set_ylabel('Non-wetting Phase Saturation')
    ax[1].step(data.pc, data.snwp, 'b-o', where='post')
    ax[1].set_title('Step diagram')
    ax[1].set_xlabel('Capillary Pressure [Pa]')
    ax[1].set_ylabel('Non-wetting Phase Saturation')
    ax[2].semilogx(data.pc, data.snwp, 'r-o')
    ax[2].set_title('Logarithmic CP')
    ax[2].set_xlabel('Capillary Pressure [Pa]')
    ax[2].set_ylabel('Non-wetting Phase Saturation')

    plt.show()
    return data


def PHASE_FRACTION(im, strel_radius=10):
    # im: binary image from Generators_3D
    # strel_radius: radius of structuring element for erosion
    # normed: if 'False' calculates the number of voxels

    im_close = fftmorphology(im=im, strel=ball(strel_radius), mode='erosion')
    pad = im*1.0 - im_close*1.0
    phases = im + pad

    labels = np.unique(phases)
    print('labels of phases are', labels)
    phase_fracs = ps.metrics.phase_fraction(phases)
    print('phase fractions', phase_fracs)
    phase_voxel_num = ps.metrics.phase_fraction(phases, normed=False)
    print('phase voxel number', phase_voxel_num)

    length = im.shape[0]
    fig, ax = plt.subplots(1, 3, figsize=[18, 6])
    ind = int(length/2)
    ax[0].imshow((im)[:, :, ind], interpolation='none')
    ax[0].set_title('Original image')
    ax[0].axis(False)
    ax[1].imshow((pad)[:, :, ind], interpolation='none')
    ax[1].set_title('Pad')
    ax[1].axis(False)
    ax[2].imshow((phases)[:, :, ind], interpolation='none')
    ax[2].set_title('Phases')
    ax[2].axis(False)

    plt.show()
    return phase_fracs


def PORE_SIZE_DISTRIBUTION(im, bins=25, voxel_size=1, log=False):
    # im: binary image from Generators_3D
    # bins: number of classes for the histogram, 10 (default)
    # voxel_size: used for scaling, 1 (default)
    # log: if False the histogram binning is based on linear(pore radius)

    im_poro = ps.filters.porosimetry(im)
    data = ps.metrics.pore_size_distribution(im=im_poro, bins=bins, voxel_size=voxel_size, log=log)

    length = im.shape[0]
    fig, ax = plt.subplots(1, 4, figsize=[20, 5])
    ind = int(length/2)
    ax[0].imshow((im_poro)[:, :, ind], interpolation='none')
    ax[0].set_title('Porosimetry image')
    ax[0].axis(False)
    ax[1].plot(data.bin_centers, data.pdf)
    ax[1].set_title("Probability Density Function")
    ax[2].plot(data.bin_centers, data.cdf)
    ax[2].set_title("Cumulative Density Function")
    ax[3].bar(data.bin_centers, data.cdf, data.bin_widths, edgecolor='k')
    ax[3].set_title('Bar Plot')

    plt.show()
    return data


def POROSITY(im):
    # im: binary image from Generators_3D

    e = ps.metrics.porosity(im)
    print(e)

    length = im.shape[0]
    fig, ax = plt.subplots(1, figsize=[8, 8])
    ind = int(length/2)
    ax.imshow(im[ind, ...], interpolation='none')
    ax.axis(False)

    plt.show()
    return e


def POROSITY_PROFILE(im, voxel_size=5.9):
    # im: binary image from Generators_3D
    # voxel_size: microns/voxel

    prfx = ps.metrics.porosity_profile(im, axis=0)
    prfy = ps.metrics.porosity_profile(im, axis=1)
    prfz = ps.metrics.porosity_profile(im, axis=2)

    plt.figure(figsize=[8, 6])
    plt.plot(np.linspace(0, im.shape[0]*voxel_size, im.shape[0]), prfx, 'b-', label='yz-plane', alpha=0.5)
    plt.plot(np.linspace(0, im.shape[1]*voxel_size, im.shape[1]), prfy, 'r-', label='xz-plane', alpha=0.5)
    plt.plot(np.linspace(0, im.shape[2]*voxel_size, im.shape[2]), prfz, 'g-', label='xy-plane', alpha=0.5)
    plt.ylabel('Porosity of slice')
    plt.xlabel('Position of slice along given axis')
    plt.legend()

    plt.figure(figsize=[8, 6])
    plt.plot(np.linspace(0, 1, im.shape[0]), prfx, 'b-', label='yz-plane', alpha=0.5)
    plt.plot(np.linspace(0, 1, im.shape[1]), prfy, 'r-', label='xz-plane', alpha=0.5)
    plt.plot(np.linspace(0, 1, im.shape[2]), prfz, 'g-', label='xy-plane', alpha=0.5)
    plt.plot([0, 1], [ps.metrics.porosity(im), ps.metrics.porosity(im)], 'r--')
    plt.ylabel('Porosity of slice')
    plt.xlabel('Fractional distance along given axis')
    plt.legend()

    plt.show()
    return prfx, prfy, prfz


def RADIAL_DENSITY_DISTRIBUTION(im, bins=50, log=True, voxel_size=0.1):
    # im: binary image from Generators_3D
    # bins: number of classes for the histogram, 10 (default)
    # log: if True the histogram binning is based on log(size)
    # voxel_size: used for scaling, 1 (default)

    dt = edt(im)
    data = ps.metrics.radial_density_distribution(dt=dt, bins=bins, log=log, voxel_size=voxel_size)

    fig, ax = plt.subplots(1, 3, figsize=[12, 4])
    ax[0].plot(data.bin_centers, data.pdf)
    ax[0].set_title("Probability Density Function")
    ax[1].plot(data.bin_centers, data.cdf)
    ax[1].set_title("Cumulative Density Function")
    ax[2].bar(data.bin_centers, data.cdf, data.bin_widths, edgecolor='k')
    ax[2].set_title('Bar Plot')

    plt.show()
    return data


def REGION_SURFACE_AREAS(im, voxel_size=0.1, strel_size=2):
    # im: binary image from Generators_3D
    # voxel_size: used for scaling, 1 (default)
    # strel: structuring element used to blur the region, bigger size means region will become increasingly blurred and inaccurate

    snow = ps.filters.snow_partitioning(im)
    regions = snow.regions
    surface_areas = ps.metrics.region_surface_areas(regions=regions, voxel_size=voxel_size, strel=ball(strel_size))

    # Labeling each region by its surface
    regions_sur = np.copy(regions)
    for i in range(0, len(surface_areas)):
        mask = np.array(np.where(regions_sur == i, regions_sur, 0), dtype=bool)
        regions_sur[mask] = surface_areas[i]

    length = im.shape[0]
    fig, ax = plt.subplots(1, 3, figsize=[12, 4])
    ind = int(length/2)
    ax[0].imshow((regions/im)[:, :, ind], interpolation='none')
    rgs = ax[1].imshow((regions_sur)[:, :, ind], interpolation='none')
    fig.colorbar(rgs, ax=ax[1], shrink=0.8)
    ax[1].axis(False)
    ax[2].hist(surface_areas, edgecolor='k')
    ax[2].set_title("PDF")

    plt.show()
    return surface_areas


def REGION_VOLUMES(im, mode='voxel'):
    # im: binary image from Generators_3D
    # mode: function that calculates the volume of each region, options: 'marching_cubes' (DEF) & 'voxel'

    snow = ps.filters.snow_partitioning(im)
    regions = snow.regions
    vols = ps.metrics.region_volumes(regions=regions, mode=mode)

    regions_vol = np.copy(regions)
    for i in range(0, len(vols)):
        mask = np.array(np.where(regions_vol == i, regions_vol, 0), dtype=bool)
        regions_vol[mask] = vols[i]

    length = im.shape[0]
    fig, ax = plt.subplots(1, 2, figsize=[10, 4])
    ind = int(length/2)
    ax[0].imshow((regions/im)[:, :, ind], interpolation='none')
    rgs = ax[1].imshow((regions_vol)[:, :, ind], interpolation='none')
    fig.colorbar(rgs, ax=ax[1], shrink=0.8)
    ax[1].axis(False)

    plt.show()
    return vols


def SATN_PROFILE(im, s_value=0.6, span=13, mode='slide', axis=1):
    # im: binary image from Generators_3D
    # s_value (ps.metrics.satn_profile): global saturation
    # span: width of slice over which the saturation is computed, default 10 voxels
    # mode: How averaging window moves, options: by sliding ('slide') or by tiling ('tile')

    inlets = np.zeros_like(im)
    inlets[0, :, :] = True
    inv = ps.simulations.drainage(im=im, inlets=inlets, voxel_size=1, g=0)
    s = ps.metrics.satn_profile(satn=inv.im_satn, s=s_value, span=span, mode=mode, axis=axis)

    length = im.shape[0]
    fig, ax = plt.subplots(1, 3, figsize=[18, 6])
    ind = int(length/2)
    ax[0].imshow((inv.im_satn/im)[:, :, ind], interpolation='none')
    ax[0].axis(False)
    ax[0].set_title('Saturation map')
    ax[1].imshow(((inv.im_satn < s_value)*(inv.im_satn > 0)/im)[:, :, ind], interpolation='none')
    ax[1].axis(False)
    ax[1].set_title(f'Fluid distribution at saturation = {s_value}')
    ax[2].plot(s.position, s.saturation, 'b-o')
    ax[2].set_xlabel("distance from injection face")
    ax[2].set_ylabel("non-wetting phase saturation")

    plt.show()
    return s
