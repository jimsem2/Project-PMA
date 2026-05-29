import matplotlib.pyplot as plt
import numpy as np
import porespy as ps
import imageio
import pandas as pd

ps.visualization.set_mpl_style()
np.random.seed(10)

def plot_results(im, res, axis):
    fig, ax = plt.subplots(1, 3, figsize=[15, 4])
    if axis is None or axis == 0:
        h = ax[0].imshow((res.im_seq / im)[int(im.shape[0]/2), :, :])
        cbar = fig.colorbar(h, aspect=10, format="{x:.0f}")
        cbar.ax.set_title("Invasion Sequence", rotation=90, y=0, c="w")
        ax[0].axis(False)
        ax[0].set_title("(a) Sequence Map")

        h = ax[1].imshow((res.im_snwp / im)[int(im.shape[0]/2), :, :])
        cbar = fig.colorbar(h, aspect=10, format="{x:.1f}")
        cbar.ax.set_title("Non-Wetting Phase Saturation", rotation=90, y=0, c="w")
        ax[1].axis(False)
        ax[1].set_title("(b) Saturation Map")

        h = ax[2].imshow(np.log10(res.im_pc)[int(im.shape[0]/2), :, :] / im[int(im.shape[0]/2), :, :])
        cbar = fig.colorbar(h, aspect=10, format="{x:.1f}")
        cbar.ax.set_title("log(Capillary Pressure)", rotation=90, y=0, c="w")
        ax[2].axis(False)
        ax[2].set_title("(c) Capillary Pressure Map")
    elif axis == 1:
        h = ax[0].imshow((res.im_seq / im)[:, int(im.shape[1]/2), :])
        cbar = fig.colorbar(h, aspect=10, format="{x:.0f}")
        cbar.ax.set_title("Invasion Sequence", rotation=90, y=0, c="w")
        ax[0].axis(False)
        ax[0].set_title("(a) Sequence Map")

        h = ax[1].imshow((res.im_snwp / im)[:, int(im.shape[1]/2), :])
        cbar = fig.colorbar(h, aspect=10, format="{x:.1f}")
        cbar.ax.set_title("Non-Wetting Phase Saturation", rotation=90, y=0, c="w")
        ax[1].axis(False)
        ax[1].set_title("(b) Saturation Map")

        h = ax[2].imshow(np.log10(res.im_pc)[:, int(im.shape[1]/2), :] / im[:, int(im.shape[1]/2), :])
        cbar = fig.colorbar(h, aspect=10, format="{x:.1f}")
        cbar.ax.set_title("log(Capillary Pressure)", rotation=90, y=0, c="w")
        ax[2].axis(False)
        ax[2].set_title("(c) Capillary Pressure Map")
    else:
        h = ax[0].imshow((res.im_seq / im)[:, :, int(im.shape[2]/2)])
        cbar = fig.colorbar(h, aspect=10, format="{x:.0f}")
        cbar.ax.set_title("Invasion Sequence", rotation=90, y=0, c="w")
        ax[0].axis(False)
        ax[0].set_title("(a) Sequence Map")

        h = ax[1].imshow((res.im_snwp / im)[:, :, int(im.shape[2]/2)])
        cbar = fig.colorbar(h, aspect=10, format="{x:.1f}")
        cbar.ax.set_title("Non-Wetting Phase Saturation", rotation=90, y=0, c="w")
        ax[1].axis(False)
        ax[1].set_title("(b) Saturation Map")

        h = ax[2].imshow(np.log10(res.im_pc)[:, :, int(im.shape[2]/2)] / im[:, :, int(im.shape[2]/2)])
        cbar = fig.colorbar(h, aspect=10, format="{x:.1f}")
        cbar.ax.set_title("log(Capillary Pressure)", rotation=90, y=0, c="w")
        ax[2].axis(False)
        ax[2].set_title("(c) Capillary Pressure Map")
    plt.show()

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
        drn = ps.simulations.drainage(im=im, pc=pc, min_size=min_size, steps=steps)
    elif axis == 0:
        inlets = ps.generators.faces(im.shape, inlet=0)
        outlets = ps.generators.faces(im.shape, outlet=0)
        drn = ps.simulations.drainage(im=im, pc=pc, min_size=min_size, inlets=inlets, outlets=outlets, steps=steps)
    elif axis == 1:
        inlets = ps.generators.faces(im.shape, inlet=1)
        outlets = ps.generators.faces(im.shape, outlet=1)
        drn = ps.simulations.drainage(im=im, pc=pc, min_size=min_size, inlets=inlets, outlets=outlets, steps=steps)
    else:
        inlets = ps.generators.faces(im.shape, inlet=2)
        outlets = ps.generators.faces(im.shape, outlet=2)
        drn = ps.simulations.drainage(im=im, pc=pc, min_size=min_size, inlets=inlets, outlets=outlets, steps=steps)

    # imageio.volsave("drainage.tif", np.array(drn.astype("uint8")))
    plot_results(im, drn, axis)

    # Plotting Saturation curve
    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(im=im, pc=drn.im_pc, mode='drainage')

    fig, ax = plt.subplots(figsize=[3.5, 3.5])
    ax.semilogx(Pc, Snwp, marker='o', markersize=6, linewidth=1, color='blue')
    ax.set_xlabel('Capillary Pressure [Pa]')
    ax.set_ylabel('Non-Wetting Phase Saturation')
    ax.set_ylim([-0.05, 1.05])
    plt.show()

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
        imb = ps.simulations.imbibition(im=im, pc=pc, min_size=min_size, steps=steps)
    elif axis == 0:
        inlets = ps.generators.faces(im.shape, inlet=0)
        outlets = ps.generators.faces(im.shape, outlet=0)
        imb = ps.simulations.imbibition(im=im, pc=pc, min_size=min_size, inlets=inlets, outlets=outlets, steps=steps)
    elif axis == 1:
        inlets = ps.generators.faces(im.shape, inlet=1)
        outlets = ps.generators.faces(im.shape, outlet=1)
        imb = ps.simulations.imbibition(im=im, pc=pc, min_size=min_size, inlets=inlets, outlets=outlets, steps=steps)
    else:
        inlets = ps.generators.faces(im.shape, inlet=2)
        outlets = ps.generators.faces(im.shape, outlet=2)
        imb = ps.simulations.imbibition(im=im, pc=pc, min_size=min_size, inlets=inlets, outlets=outlets, steps=steps)

    # imageio.volsave("imbibition.tif", np.array(imb.astype("uint8")))
    plot_results(im, imb, axis)

    # Plotting Saturation curve
    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(im=im, pc=imb.im_pc, mode='imbibition')

    fig, ax = plt.subplots(figsize=[3.5, 3.5])
    ax.semilogx(Pc, Snwp, marker='o', markersize=6, linewidth=1, color='blue')
    ax.set_xlabel('Capillary Pressure [Pa]')
    ax.set_ylabel('Non-Wetting Phase Saturation')
    ax.set_ylim([-0.05, 1.05])
    plt.show()

    if save_excel:
        pd.DataFrame({'Capillary Pressure [Pa]': Pc, 'Non-Wetting Phase Saturation': Snwp}).to_excel('Imbibition_Saturation_Curve.xlsx', index=False)
        print("Plot saved to Imbibition_Saturation_Curve.xlsx")
    return imb


def INJECTION(im, sigma=0.01, theta=180, g=9.81, rho_nwp=1000, rho_wp=0, voxel_size=1e-4, axis=None, min_size=5,
               conn='max', maxiter=500, method='qbip', save_excel=None):  # returns inj XLSX
    # im: 3D image of porous media
    # sigma: surface tension, default 0.01 N/m
    # theta: contact angle, default 180 degrees
    # g: gravitational acceleration, default 9.81 m/s^2
    # rho_nwp: non-wetting phase density, default 1000 kg/m^3
    # rho_wp: wetting phase density, default 0 kg/m^3
    # voxel_size: size of each voxel, default 1e-4 m
    # axis: axis along which imbibition occurs, 0 for x, 1 for y, 2 for z, default None (imbibition occurs from all directions)
    # min_size: the number of clustered pixels below which are ignored as trapped, default 5 voxels
    # conn:  Options: max (when a corner is considered connection between voxels), min (when only face is a connection)
    # maxiter: maximum number of iterations for the injection process, default 500
    # method: algorith used for finding the next voxel to be invaded, options: 'qbip' default (uses a binary heap, faster), 'ibip' (uses image analysis tools)

    pc = ps.filters.capillary_transform(im=im, sigma=sigma, theta=theta, g=g, rho_nwp=rho_nwp, rho_wp=rho_wp, voxel_size=voxel_size)   

    if axis is None:
        inj = ps.simulations.injection(im=im, pc=pc, min_size=min_size, conn=conn, maxiter=maxiter, method=method)
    elif axis == 0:
        inlets = ps.generators.faces(im.shape, inlet=0)
        outlets = ps.generators.faces(im.shape, outlet=0)
        inj = ps.simulations.injection(im=im, pc=pc, min_size=min_size, inlets=inlets, outlets=outlets, conn=conn, maxiter=maxiter, method=method)
    elif axis == 1:
        inlets = ps.generators.faces(im.shape, inlet=1)
        outlets = ps.generators.faces(im.shape, outlet=1)
        inj = ps.simulations.injection(im=im, pc=pc, min_size=min_size, inlets=inlets, outlets=outlets, conn=conn, maxiter=maxiter, method=method)
    else:
        inlets = ps.generators.faces(im.shape, inlet=2)
        outlets = ps.generators.faces(im.shape, outlet=2)
        inj = ps.simulations.injection(im=im, pc=pc, min_size=min_size, inlets=inlets, outlets=outlets, conn=conn, maxiter=maxiter, method=method)

    # imageio.volsave("injection.tif", np.array(inj.astype("uint8")))
    plot_results(im, inj, axis)

    # Plotting Saturation curve
    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(im=im, pc=inj.im_pc, seq=inj.im_seq, mode='drainage')

    fig, ax = plt.subplots(figsize=[3.5, 3.5])
    ax.plot(np.log10(Pc), Snwp, markersize=6, linewidth=1, color='blue')
    ax.set_xlabel('log(Capillary Pressure [Pa])')
    ax.set_ylabel('Non-Wetting Phase Saturation')
    ax.set_xlim([None, None])
    ax.set_ylim([-0.05, 1.05])
    plt.show()

    if save_excel:
        pd.DataFrame({'Capillary Pressure [Pa]': Pc, 'Non-Wetting Phase Saturation': Snwp}).to_excel('Injection_Saturation_Curve.xlsx', index=False)
        print("Plot saved to Injection_Saturation_Curve.xlsx")

    return inj


def TORTUOSITY(im, axis=0): # returns t_fd
    # im: 3D image of porous media
    # axis: axis along which tortuosity is calculated, 0 for x default, 1 for y, 2 for z
    t_fd = ps.simulations.tortuosity_fd(im, axis=axis)
    print(t_fd)
    return t_fd


