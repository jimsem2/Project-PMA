import matplotlib.pyplot as plt
import numpy as np
import porespy as ps
import imageio
import pandas as pd
from pyevtk.hl import imageToVTK

ps.visualization.set_mpl_style()
np.random.seed(10)

# Show minor tick marks on all charts produced by this module
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True


def save_results_vtk(filename, im, res):
    """Export simulation results as VTK ImageData (.vti) for ParaView visualization.

    Saves multiple fields in a single file: binary geometry, invasion sequence,
    non-wetting phase saturation, and capillary pressure.
    """
    # pyevtk expects (nz, ny, nx) C-order; porespy images are (nx, ny, nz) → transpose
    def _vtk(arr):
        return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))
    geometry = _vtk(im.astype(np.float64))
    seq = _vtk(res.im_seq.astype(np.float64))
    snwp = _vtk(res.im_snwp.astype(np.float64))
    pc = _vtk(res.im_pc.astype(np.float64))

    pointData = {
        "geometry": geometry,
        "invasion_sequence": seq,
        "saturation_nwp": snwp,
        "capillary_pressure": pc,
    }
    # imageToVTK appends .vti extension automatically
    imageToVTK(filename, pointData=pointData)
    print(f"VTK file saved to: {filename}.vti")


def plot_results(im, res, axis, title=""):
    # Helper: mask solid voxels with NaN so they don't distort colorbars
    def _pore(arr):
        return np.where(im, arr.astype(float), np.nan)

    def _log_pc():
        with np.errstate(divide='ignore', invalid='ignore'):
            lpc = np.where((res.im_pc > 0) & im, np.log10(res.im_pc), np.nan)
        return lpc

    seq = _pore(res.im_seq)
    snwp = _pore(res.im_snwp)
    lpc = _log_pc()

    fig, ax = plt.subplots(1, 3, figsize=[15, 4])
    fig.canvas.manager.set_window_title(title)

    if axis is None or axis == 0:
        s = np.s_[:, int(im.shape[1] / 2), :]
    elif axis == 1:
        s = np.s_[:, :, int(im.shape[2] / 2)]
    else:
        s = np.s_[int(im.shape[0] / 2), :, :]

    h = ax[0].imshow(seq[s])
    cbar = fig.colorbar(h, aspect=10, format="{x:.0f}")
    cbar.ax.set_title("Invasion Sequence", rotation=90, y=0, c="w")
    ax[0].axis(False)
    ax[0].set_title("(a) Sequence Map")

    h = ax[1].imshow(snwp[s])
    cbar = fig.colorbar(h, aspect=10, format="{x:.1f}")
    cbar.ax.set_title("Non-Wetting Phase Saturation", rotation=90, y=0, c="w")
    ax[1].axis(False)
    ax[1].set_title("(b) Saturation Map")

    h = ax[2].imshow(lpc[s])
    cbar = fig.colorbar(h, aspect=10, format="{x:.1f}")
    cbar.ax.set_title("log(Capillary Pressure)", rotation=90, y=0, c="w")
    ax[2].axis(False)
    ax[2].set_title("(c) Capillary Pressure Map")

    plt.show(block=False)


def DRAINAGE(im, sigma=0.01, theta=180, g=9.81, rho_nwp=1000, rho_wp=0, voxel_size=1e-4, axis=None, min_size=5, steps=None, save_excel=False):    # returns drn XLSX
    # im: 3D image of porous media
    # sigma: surface tension, default 0.01 N/m
    # theta: contact angle, default 180 degrees
    # g: gravitational acceleration, default 9.81 m/s^2
    # rho_nwp: non-wetting phase density, default 1000 kg/m^3
    # rho_wp: wetting phase density, default 0 kg/m^3
    # voxel_size: size of each voxel, default 1e-4 m
    # axis: axis along which drainage occurs, 0 for x, 1 for y, 2 for z, default None (drainage occurs from all directions)
    # min_size: the number of clustered pixels below which are ignored as trapped, default 5 voxels
    # steps: number of steps to show in the sequence map, default None (shows all steps)
    # save_excel: if True, saves the results in an Excel file, default False

    pc = ps.filters.capillary_transform(im=im, sigma=sigma, theta=theta, g=g, rho_nwp=rho_nwp, rho_wp=rho_wp, voxel_size=voxel_size)
    if axis is None:
        inlets = ps.generators.borders(shape=im.shape, mode="faces")
        drn = ps.simulations.drainage(im=im, pc=pc, min_size=min_size, inlets=inlets, steps=steps)
    elif axis == 0:
        inlets = ps.generators.faces(im.shape, inlet=0)
        outlets = ps.generators.faces(im.shape, outlet=0)
        drn = ps.simulations.drainage(im=im, pc=pc, min_size=min_size, inlets=inlets, outlets=outlets, steps=steps)
    elif axis == 1:
        inlets = ps.generators.faces(im.shape, inlet=1)
        outlets = ps.generators.faces(im.shape, outlet=1)
        drn = ps.simulations.drainage(im=im, pc=pc, min_size=min_size, inlets=inlets, outlets=outlets, steps=steps)
    elif axis == 2:
        inlets = ps.generators.faces(im.shape, inlet=2)
        outlets = ps.generators.faces(im.shape, outlet=2)
        drn = ps.simulations.drainage(im=im, pc=pc, min_size=min_size, inlets=inlets, outlets=outlets, steps=steps)
    else:
        raise ValueError(f"Unknown axis '{axis}'. Choose None, 0, 1, or 2.")


    save_results_vtk("porespy_drainage", im, drn)
    plot_results(im, drn, axis, "DRAINAGE")

    # Plotting Saturation curve
    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(im=im, pc=drn.im_pc, mode='drainage')

    fig, ax = plt.subplots(figsize=[4, 4])
    fig.canvas.manager.set_window_title("DRAINAGE - Saturation Curve")
    ax.semilogx(Pc, Snwp, marker='o', markersize=6, linewidth=1, color='blue')
    ax.set_xlabel('Capillary Pressure [Pa]')
    ax.set_ylabel('Non-Wetting Phase Saturation')
    ax.set_ylim([-0.05, 1.05])
    plt.show(block=False)

    if save_excel:
        pd.DataFrame({'Capillary Pressure [Pa]': Pc, 'Non-Wetting Phase Saturation': Snwp}).to_excel('Drainage_Saturation_Curve.xlsx', index=False)
        print("Plot saved to Drainage_Saturation_Curve.xlsx")

    return drn


def IMBIBITION(im, sigma=0.01, theta=180, g=9.81, rho_nwp=1000, rho_wp=0, voxel_size=1e-4, axis=None, min_size=5, steps=None, save_excel=False):  # returns imb XLSX
    # im: 3D image of porous media
    # sigma: surface tension, default 0.01 N/m
    # theta: contact angle, default 180 degrees
    # g: gravitational acceleration, default 9.81 m/s^2
    # rho_nwp: non-wetting phase density, default 1000 kg/m^3
    # rho_wp: wetting phase density, default 0 kg/m^3
    # voxel_size: size of each voxel, default 1e-4 m
    # axis: axis along which imbibition occurs, 0 for x, 1 for y, 2 for z, default None (imbibition occurs from all directions)
    # min_size: the number of clustered pixels below which are ignored as trapped, default 5 voxels
    # steps: number of steps to show in the sequence map, default None (shows all steps)
    # save_excel: if True, saves the results in an Excel file, default False

    pc = ps.filters.capillary_transform(im=im, sigma=sigma, theta=theta, g=g, rho_nwp=rho_nwp, rho_wp=rho_wp, voxel_size=voxel_size)
    if axis is None:
        inlets = ps.generators.borders(shape=im.shape, mode="faces")
        imb = ps.simulations.imbibition(im=im, pc=pc, min_size=min_size, inlets=inlets, steps=steps)
    elif axis == 0:
        inlets = ps.generators.faces(im.shape, inlet=0)
        outlets = ps.generators.faces(im.shape, outlet=0)
        imb = ps.simulations.imbibition(im=im, pc=pc, min_size=min_size, inlets=inlets, outlets=outlets, steps=steps)
    elif axis == 1:
        inlets = ps.generators.faces(im.shape, inlet=1)
        outlets = ps.generators.faces(im.shape, outlet=1)
        imb = ps.simulations.imbibition(im=im, pc=pc, min_size=min_size, inlets=inlets, outlets=outlets, steps=steps)
    elif axis == 2:
        inlets = ps.generators.faces(im.shape, inlet=2)
        outlets = ps.generators.faces(im.shape, outlet=2)
        imb = ps.simulations.imbibition(im=im, pc=pc, min_size=min_size, inlets=inlets, outlets=outlets, steps=steps)
    else:
        raise ValueError(f"Unknown axis '{axis}'. Choose None, 0, 1, or 2.")

    save_results_vtk("porespy_imbibition", im, imb)
    plot_results(im, imb, axis, "IMBIBITION")

    # Plotting Saturation curve
    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(im=im, pc=imb.im_pc, mode='imbibition')

    fig, ax = plt.subplots(figsize=[4, 4])
    fig.canvas.manager.set_window_title("IMBIBITION - Saturation Curve")
    ax.semilogx(Pc, Snwp, marker='o', markersize=6, linewidth=1, color='blue')
    ax.set_xlabel('Capillary Pressure [Pa]')
    ax.set_ylabel('Non-Wetting Phase Saturation')
    ax.set_ylim([-0.05, 1.05])
    plt.show(block=False)

    if save_excel:
        pd.DataFrame({'Capillary Pressure [Pa]': Pc, 'Non-Wetting Phase Saturation': Snwp}).to_excel('Imbibition_Saturation_Curve.xlsx', index=False)
        print("Plot saved to Imbibition_Saturation_Curve.xlsx")
    return imb


def INJECTION(im, sigma=0.01, theta=180, g=9.81, rho_nwp=1000, rho_wp=0, voxel_size=1e-4, axis=None, min_size=5,
               conn='max', maxiter=200, method='qbip', save_excel=None):  # returns inj XLSX
    # im: 3D image of porous media
    # sigma: surface tension, default 0.01 N/m
    # theta: contact angle, default 180 degrees
    # g: gravitational acceleration, default 9.81 m/s^2
    # rho_nwp: non-wetting phase density, default 1000 kg/m^3
    # rho_wp: wetting phase density, default 0 kg/m^3
    # voxel_size: size of each voxel, default 1e-4 m
    # axis: axis along which injection occurs, 0 for x, 1 for y, 2 for z, default None (injection occurs from all directions)
    # min_size: the number of clustered pixels below which are ignored as trapped, default 5 voxels
    # conn:  Options: max (when a corner is considered connection between voxels), min (when only face is a connection)
    # maxiter: maximum number of iterations for the injection process, default 200
    # method: algorith used for finding the next voxel to be invaded, options: 'qbip' default (uses a binary heap, faster), 'ibip' (uses image analysis tools)

    pc = ps.filters.capillary_transform(im=im, sigma=sigma, theta=theta, g=g, rho_nwp=rho_nwp, rho_wp=rho_wp, voxel_size=voxel_size)
    if axis is None:
        inlets = ps.generators.borders(shape=im.shape, mode="faces")
        inj = ps.simulations.injection(im=im, pc=pc, min_size=min_size, inlets=inlets, conn=conn, maxiter=maxiter, method=method)
    elif axis == 0:
        inlets = ps.generators.faces(im.shape, inlet=0)
        outlets = ps.generators.faces(im.shape, outlet=0)
        inj = ps.simulations.injection(im=im, pc=pc, min_size=min_size, inlets=inlets, outlets=outlets, conn=conn, maxiter=maxiter, method=method)
    elif axis == 1:
        inlets = ps.generators.faces(im.shape, inlet=1)
        outlets = ps.generators.faces(im.shape, outlet=1)
        inj = ps.simulations.injection(im=im, pc=pc, min_size=min_size, inlets=inlets, outlets=outlets, conn=conn, maxiter=maxiter, method=method)
    elif axis == 2:
        inlets = ps.generators.faces(im.shape, inlet=2)
        outlets = ps.generators.faces(im.shape, outlet=2)
        inj = ps.simulations.injection(im=im, pc=pc, min_size=min_size, inlets=inlets, outlets=outlets, conn=conn, maxiter=maxiter, method=method)
    else:
        raise ValueError(f"Unknown axis '{axis}'. Choose None, 0, 1, or 2.")

    save_results_vtk("porespy_injection", im, inj)
    plot_results(im, inj, axis, "INJECTION")

    # Plotting Saturation curve — only possible when invasion actually occurred
    Pc, Snwp = None, None
    if not np.any(inj.im_snwp > 0):
        print("Note: no pore invasion recorded — saturation curve not available.")
    else:
        try:
            try:
                Pc, Snwp = ps.metrics.pc_map_to_pc_curve(im=im, pc=inj.im_pc, seq=inj.im_seq, mode='drainage')
            except (IndexError, ValueError):
                Pc, Snwp = ps.metrics.pc_map_to_pc_curve(im=im, pc=inj.im_pc, mode='drainage')
            fig, ax = plt.subplots(figsize=[4, 4])
            fig.canvas.manager.set_window_title("INJECTION - Saturation Curve")
            ax.plot(np.log10(Pc), Snwp, markersize=6, linewidth=1, color='blue')
            ax.set_xlabel('log(Capillary Pressure [Pa])')
            ax.set_ylabel('Non-Wetting Phase Saturation')
            ax.set_xlim([None, None])
            ax.set_ylim([-0.05, 1.05])
            plt.show(block=False)
        except Exception as _e:
            print(f"Note: could not plot saturation curve ({_e})")

    if save_excel and Pc is not None:
        pd.DataFrame({'Capillary Pressure [Pa]': Pc, 'Non-Wetting Phase Saturation': Snwp}).to_excel('Injection_Saturation_Curve.xlsx', index=False)
        print("Plot saved to Injection_Saturation_Curve.xlsx")

    return inj


def TORTUOSITY(im, axis=0, crop_size=None):
    # im: 3D image of porous media
    # axis: axis along which tortuosity is calculated, 0 for x default, 1 for y, 2 for z
    # crop_size: if set, crop the centre cube of this side length before solving.
    #   tortuosity_fd allocates one network node per voxel, so images larger than
    #   ~300^3 will exhaust RAM. Set crop_size <= 300 for large images.
    if crop_size is not None:
        slices = tuple(
            slice(max(0, (s - crop_size) // 2),
                  max(0, (s - crop_size) // 2) + min(s, crop_size))
            for s in im.shape)
        im = im[slices]
        print(f"Image cropped to {im.shape} for tortuosity calculation.")
    t_fd = ps.simulations.tortuosity_fd(im, axis=axis)
    print(t_fd)
    return t_fd

