import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import porespy as ps
import scipy.ndimage as spim
import imageio
from porespy.filters import fftmorphology
from skimage.morphology import ball
from edt import edt

ps.visualization.set_mpl_style()
np.random.seed(10)

# Show minor tick marks on all charts produced by this module
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True

# YELLOW = TRUE = 1 = VOID/PORE // DARK BLUE = FALSE = 0 = SOLID

def BOND_NUMBER(im, g=9.81, sigma=0.01, delta_rho=1000, voxel_size=1, source='lt', method='median'): # returns bn
    # im: binary image from Generators_3D
    # g: gravitational acceleration, default 9.81 m/s^2
    # sigma: surface tension, default 0.01 N/m
    # delta_rho: density difference, default 1000 kg/m^3
    # voxel_size: length in meters of voxel edges, used for scaling, 1 (default)
    # source: how the characteristic size is calculated, default 'lt', options: 'dt' for distance transform, 'lt' for local thickness
    # method: how the values from 'source'are averaged, default 'median', options: 'mean', 'median', 'min', 'max'

    bn = ps.metrics.bond_number(im=im, g=g, sigma=sigma, delta_rho=delta_rho, voxel_size=voxel_size, source=source, method=method)
    print(f'Bond number is {bn:.7f}')

    return bn


def CHORD_COUNTS(im, spacing=1, trim_edges=True, apply_chords3D=True, axis=0, bins=25): # returns cc
    # Applies chords and counts their lengths. Works both with apply_chords and apply_chords_3D
    # spacing: space between chords, trim_edges: to remove edges that touch borders
    # apply_chords3D: if True applies apply_chords_3D, if False applies apply_chords
    # axis: the axis along which chords are drawn, only used when apply_chords3D is False, options: 0 for x axis, 1 for y axis, 2 for z axis
    # bins: number of classes for the histogram, 25 (default)

    if apply_chords3D:
        chords = ps.filters.apply_chords_3D(im=im, spacing=spacing, trim_edges=trim_edges)
        cc = ps.metrics.chord_counts(im=chords)

        fig, ax = plt.subplots(1, figsize=[6, 6])
        fig.canvas.manager.set_window_title("CHORD COUNTS")
        ax.hist(cc, bins=bins)
        ax.set_title("CHORD COUNTS")
        ax.set_xlabel('Chord Length')
        ax.set_ylabel('Frequency')
    else:
        chords = ps.filters.apply_chords(im=im, spacing=spacing, axis=axis, trim_edges=trim_edges, label=True)
        cc = ps.metrics.chord_counts(im=chords)

        fig, ax = plt.subplots(1, figsize=[6, 6])
        fig.canvas.manager.set_window_title("CHORD COUNTS")
        ax.hist(cc, bins=bins)
        ax.set_title("CHORD COUNTS")
        ax.set_xlabel('Chord Length Along Selected Axis')
        ax.set_ylabel('Frequency')

    plt.show(block=False)
    return cc


def CHORD_LENGTH_DISTRIBUTION(im, bins=25, log=False, voxel_size=1, normalization='number', apply_chords3D=True, axis=0, save_excel=False):   # returns cld_data  XLSX
    # im: binary image from Generators_3D
    # bins: number of classes for the histogram, 25 (default)
    # log: logarithmic scale, if True then 'data.L' must be 'data.LogL' in the diagrams
    # voxel_size: length in meters of voxel edges, used for scaling, 1 (default)
    # normalization: how to normalize bin heights, options: 'count' or 'number' (default), 'length'
    # axis: the axis along which chords are drawn, only used when apply_chords3D is False, options: 0 for x axis, 1 for y axis, 2 for z axis

    if apply_chords3D:
        im_chords = ps.filters.apply_chords_3D(im=im)
    else:
        im_chords = ps.filters.apply_chords(im=im, axis=axis)
    cld_data = ps.metrics.chord_length_distribution(im=im_chords, bins=bins, log=log, voxel_size=voxel_size, normalization=normalization)

    fig, ax = plt.subplots(1, 3, figsize=[12, 4])
    fig.canvas.manager.set_window_title("CHORD LENGTH DISTRIBUTION")
    ax[0].plot(cld_data.L, cld_data.pdf)
    ax[0].set_title("PROBABILITY DENSITY FUNCTION")
    ax[0].set_xlabel("Chord Length")
    ax[0].set_ylabel("Probability Density")
    ax[1].plot(cld_data.L, cld_data.cdf)
    ax[1].set_title("CUMULATIVE DENSITY FUNCTION")
    ax[1].set_xlabel("Chord Length")
    ax[1].set_ylabel("Cumulative Density")
    ax[2].bar(cld_data.L, cld_data.cdf, cld_data.bin_widths, edgecolor='k')
    ax[2].set_title('BAR PLOT')
    ax[2].set_xlabel("Chord Length")
    ax[2].set_ylabel("Frequency")

    plt.show(block=False)

    if save_excel:
        pd.DataFrame({'L': cld_data.L, 'pdf': cld_data.pdf, 'cdf': cld_data.cdf}).to_excel('chord_length_distribution.xlsx', index=False)
        print("Plot saved to chord_length_distribution.xlsx")

    return cld_data


def FIND_H(im, srange=[0.01, 0.95], axis_of_intrusion=0, voxel_size=1, g=0, s=0.5): # returns h
    # im: binary image from Generators_3D
    # srange: saturation range to apply find_h. It will start searching from the highest saturation value in srange and will stop at the lowest saturation value in srange. Default: [0.01, 0.95]
    # axis_of_intrusion: the axis along which the intrusion happens, options: 0 for x axis, 1 for y axis, 2 for z axis
    # voxel_size: length in meters of voxel edges, used for scaling, 1 (default)
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
    fig.canvas.manager.set_window_title("FIND H")
    ax.plot(s_prof.position, s_prof.saturation, 'b-o')
    ax.set_title("FIND H")
    ax.set_xlabel("Distance From Injection Face")
    ax.set_ylabel("Non-Wetting Phase Saturation")
   
    h = ps.metrics.find_h(s_prof.saturation, s_prof.position, srange=srange)
    print(h)

    plt.show(block=False)
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


def LINEAL_PATH_DISTRIBUTION(im, bins=25, log=False, voxel_size=1, save_excel=False): # returns lpd_data  XLSX
    # im: binary image from Generators_3D
    # bins: number of classes for the histogram, 25 (default)
    # log: logarithmic scale, if True then 'data.L' must be 'data.LogL' in the diagrams
    # voxel_size: length in meters of voxel edges, used for scaling, 1 (default)

    im_lin = ps.filters.distance_transform_lin(im=im, axis=0, mode='forward')
    lpd_data = ps.metrics.lineal_path_distribution(im_lin, bins=bins, log=log, voxel_size=voxel_size)

    x = lpd_data.LogL if log else lpd_data.L

    fig, ax = plt.subplots(1, 3, figsize=[10, 4])
    fig.canvas.manager.set_window_title("LINEAL PATH DISTRIBUTION")
    ax[0].plot(x, lpd_data.pdf)
    ax[0].set_title("PROBABILITY DENSITY FUNCTION")
    ax[0].set_xlabel("Lineal Path Length")
    ax[0].set_ylabel("Probability Density")
    ax[1].plot(x, lpd_data.cdf)
    ax[1].set_title("CUMULATIVE DENSITY FUNCTION")
    ax[1].set_xlabel("Lineal Path Length")
    ax[1].set_ylabel("Cumulative Density")
    ax[2].bar(x, lpd_data.cdf, lpd_data.bin_widths, edgecolor='k')
    ax[2].set_title('BAR PLOT')
    ax[2].set_xlabel("Lineal Path Length")
    ax[2].set_ylabel("Frequency")

    plt.show(block=False)

    if save_excel:
        pd.DataFrame({'L': x, 'pdf': lpd_data.pdf, 'cdf': lpd_data.cdf}).to_excel('lineal_path_distribution.xlsx', index=False)
        print("Plot saved to lineal_path_distribution.xlsx")

    return lpd_data


def MESH_SURFACE_AREA(im, voxel_size=1): # returns surface_area
    # im: binary image from Generators_3D
    # voxel_size: length in meters of voxel edges, used for scaling, 1 (default)

    mesh = ps.tools.mesh_region(region=im)
    surface_area = ps.metrics.mesh_surface_area(mesh=mesh)
    print(f'Surface area of void space is {surface_area:.5f} voxel_units^2')
    print(f'Surface area of void space is {surface_area * voxel_size**2:.10f} m^2')

    return surface_area


def MESH_VOLUME(im, voxel_size=1):   # returns volume
    # im: binary image from Generators_3D
    # voxel_size: length in meters of voxel edges, used for scaling, 1 (default)

    volume = ps.metrics.mesh_volume(region=im)
    print(f'Total volume is = {volume:.5f} voxel_units^3')
    print(f'Total volume is = {volume * voxel_size**3:.10f} m^3')

    return volume


def PC_MAP_TO_PC_CURVE(im, sigma=0.01, theta=180, voxel_size=1, steps=25, save_excel=False): # returns pc_curve   XLSX
    # im: binary image from Generators_3D
    # sigma: surface tension of the fluid-fluid system of interest
    # theta: contact angle measured through the invading phase in degrees
    # voxel_size: length in meters of voxel edges, used for scaling, 1 (default)
    # steps: number of different capillary pressures to test, default 25

    pc = ps.filters.capillary_transform(im=im, sigma=sigma, theta=theta, voxel_size=voxel_size)
    drn = ps.simulations.drainage(im=im, pc=pc, steps=steps)
    pc_curve = ps.metrics.pc_map_to_pc_curve(im=im, pc=drn.im_pc)

    print(pc_curve)

    fig, ax = plt.subplots(1, 3, figsize=[15, 5])
    fig.canvas.manager.set_window_title("PC MAP TO PC CURVE")
    ax[0].plot(pc_curve.pc, pc_curve.snwp, 'b-o')
    ax[0].set_title('PLOT DIAGRAM')
    ax[0].set_xlabel('Capillary Pressure [Pa]')
    ax[0].set_ylabel('Non-Wetting Phase Saturation')
    ax[1].step(pc_curve.pc, pc_curve.snwp, 'b-o', where='post')
    ax[1].set_title('STEP DIAGRAM')
    ax[1].set_xlabel('Capillary Pressure [Pa]')
    ax[1].set_ylabel('Non-Wetting Phase Saturation')
    ax[2].semilogx(pc_curve.pc, pc_curve.snwp, 'r-o')
    ax[2].set_title('LOGARITHMIC CP')
    ax[2].set_xlabel('Capillary Pressure [Pa]')
    ax[2].set_ylabel('Non-Wetting Phase Saturation')

    plt.show(block=False)

    if save_excel:
        pd.DataFrame({'pc': pc_curve.pc, 'snwp': pc_curve.snwp}).to_excel('pc_map_to_pc_curve.xlsx', index=False)
        print("Plot saved to pc_map_to_pc_curve.xlsx")

    return pc_curve


def PERCOLATING_POROSITY(im, conn="max"):   # returns eps0, eps1, eps2
    # im: binary image from Generators_3D
    # conn:  Options: max (default, when a corner is considered connection between voxels), min (when only face is a connection)

    eps0 = ps.metrics.percolating_porosity(im=im, conn=conn, axis=0)
    eps1 = ps.metrics.percolating_porosity(im=im, conn=conn, axis=1)
    eps2 = ps.metrics.percolating_porosity(im=im, conn=conn, axis=2)
    print(f'percolating porosity in x-axis is {eps0:.4f}')
    print(f'percolating porosity in y-axis is {eps1:.4f}')
    print(f'percolating porosity in z-axis is {eps2:.4f}')

    return eps0, eps1, eps2


def PORE_SIZE_DISTRIBUTION(im, bins=25, voxel_size=1, log=False, save_excel=False):   # returns psd_data  XLSX
    # im: binary image from Generators_3D
    # bins: number of classes for the histogram, 25 (default)
    # voxel_size: length in meters of voxel edges, used for scaling, 1 (default)
    # log: if False the histogram binning is based on linear(pore radius)

    im_poro = ps.filters.porosimetry(im=im)
    psd_data = ps.metrics.pore_size_distribution(im=im_poro, bins=bins, voxel_size=voxel_size, log=log)

    fig, ax = plt.subplots(1, 3, figsize=[15, 5])
    fig.canvas.manager.set_window_title("PORE SIZE DISTRIBUTION")
    ax[0].plot(psd_data.bin_centers, psd_data.pdf)
    ax[0].set_title("PROBABILITY DENSITY FUNCTION")
    ax[0].set_xlabel('Pore Radius In Voxels')
    ax[0].set_ylabel('Frequency')
    ax[1].plot(psd_data.bin_centers, psd_data.cdf)
    ax[1].set_title("CUMULATIVE DENSITY FUNCTION")
    ax[1].set_xlabel('Pore Radius In Voxels')
    ax[1].set_ylabel('Cumulative Frequency')
    ax[2].bar(psd_data.bin_centers, psd_data.cdf, psd_data.bin_widths, edgecolor='k')
    ax[2].set_title('BAR PLOT')
    ax[2].set_xlabel('Pore Radius In Voxels')
    ax[2].set_ylabel('Cumulative Frequency')

    plt.show(block=False)

    if save_excel:
        pd.DataFrame({'bin_centers': psd_data.bin_centers, 'pdf': psd_data.pdf, 'cdf': psd_data.cdf}).to_excel('pore_size_distribution.xlsx', index=False)
        print("Plot saved to pore_size_distribution.xlsx")

    return psd_data


def POROSITY(im):   # returns porosity
    # im: binary image from Generators_3D

    porosity = ps.metrics.porosity(im=im)
    print(f"Porosity (void fraction): {porosity:.5f}")

    return porosity


def POROSITY_PROFILE(im, voxel_size=1):   # returns prfx, prfy, prfz
    # im: binary image from Generators_3D
    # voxel_size: microns/voxel, default 1

    prfx = ps.metrics.porosity_profile(im=im, axis=0)
    prfy = ps.metrics.porosity_profile(im=im, axis=1)
    prfz = ps.metrics.porosity_profile(im=im, axis=2)

    fig, ax = plt.subplots(1, figsize=[6, 6])
    fig.canvas.manager.set_window_title("POROSITY PROFILE")
    ax.plot(prfx.position * voxel_size, prfx.porosity, 'b-', label='yz-plane', alpha=0.5)
    ax.plot(prfy.position * voxel_size, prfy.porosity, 'r-', label='xz-plane', alpha=0.5)
    ax.plot(prfz.position * voxel_size, prfz.porosity, 'g-', label='xy-plane', alpha=0.5)
    ax.plot([0, 1], [ps.metrics.porosity(im=im), ps.metrics.porosity(im=im)], 'r--')
    ax.set_title("POROSITY PROFILE")
    ax.set_ylabel('Porosity Of Slice')
    ax.set_xlabel('Position Of Slice Along Given Axis')
    ax.legend()

    fig, ax = plt.subplots(1, figsize=[6, 6])
    fig.canvas.manager.set_window_title("POROSITY PROFILE - Normalised")
    ax.plot(prfx.position / prfx.position.max(), prfx.porosity, 'b-', label='yz-plane', alpha=0.5)
    ax.plot(prfy.position / prfy.position.max(), prfy.porosity, 'r-', label='xz-plane', alpha=0.5)
    ax.plot(prfz.position / prfz.position.max(), prfz.porosity, 'g-', label='xy-plane', alpha=0.5)
    ax.plot([0, 1], [ps.metrics.porosity(im=im), ps.metrics.porosity(im=im)], 'r--')
    ax.set_title("POROSITY PROFILE")
    ax.set_ylabel('Porosity Of Slice')
    ax.set_xlabel('Fractional Distance Along Given Axis')
    ax.legend()

    plt.show(block=False)
    return prfx, prfy, prfz


def RADIAL_DENSITY_DISTRIBUTION(im, bins=25, log=False, voxel_size=1, save_excel=False):  # returns rdd_data  XLSX
    # im: binary image from Generators_3D
    # bins: number of classes for the histogram, 25 (default)
    # log: if True the histogram binning is based on log(size)
    # voxel_size: length in meters of voxel edges, used for scaling, 1 (default)

    dt = edt(im)
    rdd_data = ps.metrics.radial_density_distribution(dt=dt, bins=bins, log=log, voxel_size=voxel_size)

    fig, ax = plt.subplots(1, 3, figsize=[12, 4])
    fig.canvas.manager.set_window_title("RADIAL DENSITY DISTRIBUTION")
    ax[0].plot(rdd_data.bin_centers, rdd_data.pdf)
    ax[0].set_title("PROBABILITY DENSITY FUNCTION")
    ax[0].set_xlabel("Radial Distance")
    ax[0].set_ylabel("Probability Density")
    ax[1].plot(rdd_data.bin_centers, rdd_data.cdf)
    ax[1].set_title("CUMULATIVE DENSITY FUNCTION")
    ax[1].set_xlabel("Radial Distance")
    ax[1].set_ylabel("Cumulative Density")
    ax[2].bar(rdd_data.bin_centers, rdd_data.cdf, rdd_data.bin_widths, edgecolor='k')
    ax[2].set_title('BAR PLOT')
    ax[2].set_xlabel("Radial Distance")
    ax[2].set_ylabel("Frequency")

    plt.show(block=False)

    if save_excel:
        pd.DataFrame({'bin_centers': rdd_data.bin_centers, 'pdf': rdd_data.pdf, 'cdf': rdd_data.cdf}).to_excel('radial_density_distribution.xlsx', index=False)
        print("Plot saved to radial_density_distribution.xlsx")

    return rdd_data


def REGION_SURFACE_AREAS(im, voxel_size=1, strel_size=2, slice=0, save_excel=False):   # returns surface_areas, regions_sur XLSX
    # im: binary image from Generators_3D
    # voxel_size: length in meters of voxel edges, used for scaling, 1 (default)
    # strel_size: size of the structuring element used to blur the region, bigger size means region will become increasingly blurred and inaccurate
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
    fig.canvas.manager.set_window_title("REGION SURFACE AREAS")
    ind = int(im.shape[slice]/2)
    if slice == 0:
        im_plot = ax[0].imshow((regions_sur)[ind, :, :], interpolation='none')
        ax[0].set_title('YZ SLICE - SURFACE AREA OF REGIONS')
    elif slice == 1:
        im_plot = ax[0].imshow((regions_sur)[:, ind, :], interpolation='none')
        ax[0].set_title('XZ SLICE - SURFACE AREA OF REGIONS')
    else:
        im_plot = ax[0].imshow((regions_sur)[:, :, ind], interpolation='none')
        ax[0].set_title('XY SLICE - SURFACE AREA OF REGIONS')
    fig.colorbar(im_plot, ax=ax[0], shrink=0.8)
    ax[0].axis(False)

    ax[1].hist(surface_areas, edgecolor='k')
    ax[1].set_title("PROBABILITY DENSITY FUNCTION")
    ax[1].set_xlabel("Surface Area")
    ax[1].set_ylabel("Frequency")

    plt.show(block=False)

    if save_excel:
        pd.DataFrame({'region': range(1, len(surface_areas) + 1), 'surface_area': surface_areas}).to_excel('region_surface_areas.xlsx', index=False)
        print("Table saved to region_surface_areas.xlsx")

    return surface_areas, regions_sur


def REGION_VOLUMES(im, method='voxel', slice=0, save_excel=False):    # returns r_volumes, regions_vol XLSX
    # im: binary image from Generators_3D
    # method: how the volume of each region is calculated, options: 'marching_cubes' (DEF) & 'voxel'
    # slice: according to which axis to show the plane of the image, 0 for x, 1 for y, 2 for z

    snow = ps.filters.snow_partitioning(im=im)
    regions = snow.regions
    r_volumes = ps.metrics.region_volumes(regions=regions, method=method)

        # Assigning each region its volume
    regions_vol = np.copy(regions)
    for label, volume in enumerate(r_volumes, start=1):
        mask = regions == label
        regions_vol[mask] = volume

    fig, ax = plt.subplots(1, 2, figsize=[12, 5])
    fig.canvas.manager.set_window_title("REGION VOLUMES")
    ind = int(im.shape[slice]/2)
    if slice == 0:
        rgs = ax[0].imshow((regions_vol)[ind, :, :], interpolation='none')
        ax[0].set_title('YZ SLICE - VOLUME OF REGIONS')
    elif slice == 1:
        rgs = ax[0].imshow((regions_vol)[:, ind, :], interpolation='none')
        ax[0].set_title('XZ SLICE - VOLUME OF REGIONS')
    else:
        rgs = ax[0].imshow((regions_vol)[:, :, ind], interpolation='none')
        ax[0].set_title('XY SLICE - VOLUME OF REGIONS')
    fig.colorbar(rgs, ax=ax[0], shrink=0.8)
    ax[0].axis(False)

    ax[1].hist(r_volumes, edgecolor='k')
    ax[1].set_title("PROBABILITY DENSITY FUNCTION")
    ax[1].set_xlabel("Volume")
    ax[1].set_ylabel("Frequency")

    plt.show(block=False)

    if save_excel:
        pd.DataFrame({'region': range(1, len(r_volumes) + 1), 'volume': r_volumes}).to_excel('region_volumes.xlsx', index=False)
        print("Table saved to region_volumes.xlsx")

    return r_volumes, regions_vol


def REGIONPROPS_3D(im, sphericity=False, solidity=False, num_pixels=False):   # returns props   XLSX
    # im: binary image from Generators_3D
    # sphericity: if True, saves an image of the sphericity of each region, default False
    # solidity: if True, saves an image of the solidity of each region, default False
    # num_pixels: if True, saves an image of the number of pixels in each region, default False

    regions = spim.label(im)[0]     # to label the image for regionprops, snow partitioning is not used because it can assign the same label to different regions if they are not connected in the image, while spim.label will assign different labels to all disconnected regions in the image
    props = ps.metrics.regionprops_3D(regions)

    df = ps.metrics.props_to_DataFrame(props)

        # Image generation for visualization of selected properties
    if sphericity:
        im_sph = ps.metrics.prop_to_image(props, im.shape, "sphericity")
        imageio.volsave("im_sph.tif", np.array(im_sph.astype("uint8")))
    if solidity:
        im_solid = ps.metrics.prop_to_image(props, im.shape, "solidity")
        imageio.volsave("im_solid.tif", np.array(im_solid.astype("uint8")))
    if num_pixels:
        im_pixels = ps.metrics.prop_to_image(props, im.shape, "num_pixels")
        imageio.volsave("im_pixels.tif", np.array(im_pixels.astype("uint8")))

    def _fmt(x):
        if isinstance(x, float):
            if np.isinf(x) or np.isnan(x):
                return str(x)
            if x == int(x):
                return f'{x:.2f}'
            return f'{x:.4f}'
        return str(x)

    df_str = df.map(_fmt)

    df_str.to_excel('regionprops_3D.xlsx', index=True)
    print("Table saved to regionprops_3D.xlsx")

    return props


def REV_POROSITY(im, n=1000):  # returns rev_porosity
    # im: binary image from Generators_3D
    # n: number of randomly located sized boxes, default 1000

    rev_porosity = ps.metrics.rev_porosity(im=im, n=n)

    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    fig.canvas.manager.set_window_title("REV POROSITY")
    ax.plot(rev_porosity.volume, rev_porosity.porosity, "r.")
    ax.set_title("REV POROSITY")
    ax.set_xlabel("Volume")
    ax.set_ylabel("Porosity")

    plt.show(block=False)
    return rev_porosity


def SATN_PROFILE(im, span=2, mode='tile', intr_axis=0, sigma=0.01, theta=180, g=0,
                 s_min=0.1, s_max=0.9, s_step=0.2):  # returns sp_list
    # im: binary image from Generators_3D
    # span: width of slice over which the saturation is computed, default 2 voxels
    # mode: How averaging window moves, options: by sliding ('slide') or by tiling ('tile')
    # intr_axis: the axis along which the intrusion happens, options: 0 for x axis, 1 for y axis, 2 for z axis
    # sigma: surface tension of the fluid-fluid system of interest
    # theta: contact angle measured through the invading phase in degrees
    # g: gravitational acceleration, 0 (default)
    # s_min / s_max / s_step: saturation thresholds to profile — one curve is
    #   drawn per value of np.arange(s_min, s_max + s_step/2, s_step).
    #   Defaults reproduce the original 0.1, 0.3, 0.5, 0.7, 0.9 series.

    inlets = np.zeros_like(im)
    if intr_axis == 0:
        inlets[0, :, :] = True
    elif intr_axis == 1:
        inlets[:, 0, :] = True
    else:
        inlets[:, :, 0] = True

    pc = ps.filters.capillary_transform(im=im, sigma=sigma, theta=theta, g=g)
    inv = ps.simulations.drainage(im=im, pc=pc, inlets=inlets)

    # Thresholds above the maximum saturation actually reached would make
    # ps.metrics.satn_profile raise — report and skip them instead.
    satn_max = float(inv.im_snwp[im].max()) if im.any() else 0.0
    s_values = np.arange(s_min, s_max + s_step / 2, s_step)
    usable = [s for s in s_values if s <= satn_max]
    skipped = [s for s in s_values if s > satn_max]
    if skipped:
        print(f"Max saturation reached in the image is {satn_max:.3f}; "
              "skipped threshold(s): "
              + ", ".join(f"{s:.2f}" for s in skipped))
    if not usable:
        raise ValueError(
            f"All thresholds exceed the maximum saturation ({satn_max:.3f}) "
            "reached in the image. Lower 's_min' (and 's_max') below that value.")

    fig, ax = plt.subplots(1, figsize=[6, 6])
    fig.canvas.manager.set_window_title("SATN PROFILE")
    sp_list = []
    for s_value in usable:
        sp = ps.metrics.satn_profile(satn=inv.im_snwp, s=s_value, span=span, mode=mode, axis=intr_axis)
        sp_list.append(sp)
        ax.plot(sp.position, sp.saturation, '-o', label=f's={s_value:.1f}')
    ax.set_title("SATURATION PROFILE")
    ax.set_xlabel("Distance From Injection Face")
    ax.set_ylabel("Non-Wetting Phase Saturation")
    ax.legend()

    plt.show(block=False)
    return sp_list

