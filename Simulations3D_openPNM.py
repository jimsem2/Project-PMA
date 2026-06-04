from tokenize import Name

import matplotlib.pyplot as plt
import numpy as np
import porespy as ps
import imageio
import pandas as pd
import openpnm as op
import os

length = 60
im = ps.generators.blobs(shape=[length, length, length], porosity=0.6, blobiness=2, seed=10)
im_snow2 = ps.networks.snow2(im, voxel_size=1)
pn = op.io.network_from_porespy(im_snow2.network)

# Trim disconnected clusters so transport algorithms aren't ill-conditioned.
health = op.utils.check_network_health(pn)
if len(health['disconnected_pores']) > 0:
    op.topotools.trim(network=pn, pores=health['disconnected_pores'])
    print(f"Trimmed {len(health['disconnected_pores'])} disconnected pores.")

# Add geometric size-factor models required by Hagen-Poiseuille / Fickian conductance models.
# Networks extracted from PoreSpy don't include these by default and use different
# naming for diameter / length, so we first create the expected aliases.
# Use pyramids_and_cuboids since it does not require throat diameter < pore diameter.
pn['pore.diameter'] = pn['pore.equivalent_diameter']
pn['throat.diameter'] = pn['throat.inscribed_diameter']
pn['throat.length'] = pn['throat.total_length']
pn.add_model(propname='throat.hydraulic_size_factors',
             model=op.models.geometry.hydraulic_size_factors.pyramids_and_cuboids)
pn.add_model(propname='throat.diffusive_size_factors',
             model=op.models.geometry.diffusive_size_factors.pyramids_and_cuboids)
# Throat volume (cylinder approx.) is needed by Drainage.pc_curve()
pn.add_model(propname='throat.volume',
             model=op.models.geometry.throat_volume.cylinder)
pn.regenerate_models()

output_dir = "results_paraview"
os.makedirs(output_dir, exist_ok=True)

# Axis labels mapping: axis -> (inlet face, outlet face, coordinate index)
# PoreSpy networks label boundary pores as xmin/xmax/ymin/ymax/zmin/zmax.
AXIS_MAP = {
    "x": ("xmin", "xmax", 0),
    "y": ("ymin", "ymax", 1),
    "z": ("zmin", "zmax", 2),
}


def get_axis_faces(axis):
    # Return (inlet_label, outlet_label, coord_index) for a given axis

    axis = axis.strip().lower()
    if axis not in AXIS_MAP:
        raise ValueError(f"Unknown axis '{axis}'. Choose 'x', 'y', or 'z'.")
    return AXIS_MAP[axis]


# Create and add a phase to the network

def select_phase(network, phase_name='water'):  # returns phase
    # network: the pore network to which the phase will be added
    # phase_name: string indicating the type of phase ('water', 'air', 'mercury')

    phase_name = phase_name.strip().lower()
    if phase_name == "water":
        phase = op.phase.Water(network=network, name="water")
        # OpenPNM's Water phase doesn't define diffusivity by default.
        # Use water self-diffusion coefficient at room temperature.
        phase["pore.diffusivity"] = 2.3e-9  # m^2/s
        phase["pore.contact_angle"] = 110.0  # degrees, water on hydrophobic substrate
    elif phase_name == "air":
        phase = op.phase.Air(network=network, name="air")
        # Air phase lacks surface tension and contact angle. Use water-air
        # interfacial tension (~0.072 N/m) and assume air is the non-wetting
        # phase with contact angle 0° (measured through the invading phase).
        phase["pore.surface_tension"] = 0.072  # N/m
        phase["pore.contact_angle"] = 180.0    # degrees (non-wetting)
        phase["pore.diffusivity"] = 2.0e-5     # m^2/s (O2 in air approx.)
    elif phase_name == "mercury":
        phase = op.phase.Mercury(network=network, name="mercury")
        # Mercury phase doesn't define diffusivity by default.
        phase["pore.diffusivity"] = 1.6e-9  # m^2/s (Hg self-diffusion approx.)
        # Mercury intrusion: surface tension ~0.485 N/m, contact angle ~140°
        phase["pore.surface_tension"] = 0.485
        phase["pore.contact_angle"] = 140.0
    else:
        raise ValueError(f"Unknown phase '{phase_name}'. Choose 'water', 'air', or 'mercury'.")

    # Interpolate pore-scale properties to throats (needed by physics models)
    phase.add_model(
        propname="throat.diffusivity",
        model=op.models.misc.from_neighbor_pores,
        prop="pore.diffusivity",
        mode="min")
    phase.add_model(
        propname="throat.surface_tension",
        model=op.models.misc.from_neighbor_pores,
        prop="pore.surface_tension",
        mode="min")
    phase.add_model(
        propname="throat.contact_angle",
        model=op.models.misc.from_neighbor_pores,
        prop="pore.contact_angle",
        mode="min")

    phase.add_model(
        propname="throat.hydraulic_conductance",
        model=op.models.physics.hydraulic_conductance.hagen_poiseuille)
    phase.add_model(
        propname="throat.diffusive_conductance",
        model=op.models.physics.diffusive_conductance.ordinary_diffusion)
    phase.add_model(
        propname="throat.entry_pressure",
        model=op.models.physics.capillary_pressure.washburn)
    phase.regenerate_models()
    print(f"Phase '{phase_name}' added to the network.")
    return phase

babis = select_phase(pn, phase_name="water")

def simulate_stokes_flow(network, phase, axis="x", inlet_pressure=1.0, outlet_pressure=0.0):
    # Simulate Stokes flow to calculate pressure gradients, velocity distributions, and Darcy permeability
    # axis: The axis along which the pressure drop is applied ('x', 'y', or 'z')
    # Results are saved as VTK for ParaView

    inlet_label, outlet_label, coord_idx = get_axis_faces(axis)
    sf = op.algorithms.StokesFlow(network=network, phase=phase)

    inlet_pores = network.pores(inlet_label)
    outlet_pores = network.pores(outlet_label)

    sf.set_value_BC(pores=inlet_pores, values=inlet_pressure)
    sf.set_value_BC(pores=outlet_pores, values=outlet_pressure)
    sf.run()

    # Extract pressure field
    pressure = sf["pore.pressure"]
    phase["pore.pressure"] = pressure

    # Calculate flow rate through the outlet
    flow_rate = sf.rate(pores=outlet_pores, mode="group")       # group mode sums the flow through all outlet pores
    print(f"Total flow rate through outlet ({axis}-axis): {flow_rate[0]:.6e}")

    # Calculate velocity at throats (Q / A)
    throat_area = network["throat.cross_sectional_area"]
    throat_flow = sf.rate(throats=network.Ts, mode="single")    # single mode gives flow through each throat, Ts: T for throats, s for indices of all throats
    velocity = np.abs((throat_flow) / throat_area)
    phase["throat.velocity"] = velocity

    # Calculate Darcy permeability: K = Q * mu * L / (A * dP)
    mu = phase["pore.viscosity"].mean()
    L = np.ptp(network["pore.coords"][:, coord_idx])  # length along chosen axis
    # Cross-sectional area from the other two axes
    other_axes = [i for i in range(3) if i != coord_idx]
    L1 = np.ptp(network["pore.coords"][:, other_axes[0]])
    L2 = np.ptp(network["pore.coords"][:, other_axes[1]])
    A = L1 * L2
    delta_P = inlet_pressure - outlet_pressure  # applied pressure difference
    K = np.abs(flow_rate[0]) * mu * L / (A * delta_P)
    print(f"Darcy permeability: {K:.6e} m^2")

    # Save to VTK for ParaView
    projSF = network.project
    op.io.project_to_vtk(project=projSF, filename=os.path.join(output_dir, "stokes_flow"), fill_nans=0.0)
    print(f"Stokes flow results saved to '{output_dir}/stokes_flow.vtp'")

    return sf, K

simulate_stokes_flow(pn, babis, axis="x")

def simulate_fickian_diffusion(network, phase, axis="x"):
    """
    Simulate Fickian diffusion to calculate concentration distributions,
    effective diffusivity, and tortuosity.

    Parameters
    ----------
    axis : str
        The axis along which the concentration gradient is applied ('x', 'y', or 'z').

    Results are saved as VTK for ParaView.
    """
    inlet_label, outlet_label, coord_idx = get_axis_faces(axis)
    fd = op.algorithms.FickianDiffusion(network=network, phase=phase)

    inlet_pores = network.pores(inlet_label)
    outlet_pores = network.pores(outlet_label)

    fd.set_value_BC(pores=inlet_pores, values=1.0)
    fd.set_value_BC(pores=outlet_pores, values=0.0)
    fd.run()

    # Extract concentration field
    concentration = fd["pore.concentration"]
    phase["pore.concentration"] = concentration

    # Calculate diffusive rate
    diff_rate = fd.rate(pores=outlet_pores, mode="group")
    print(f"Total diffusive rate through outlet ({axis}-axis): {diff_rate[0]:.6e}")

    # Calculate effective diffusivity: D_eff = J * L / (A * dC)
    D_bulk = phase["pore.diffusivity"].mean()
    L = np.ptp(network["pore.coords"][:, coord_idx])
    other_axes = [i for i in range(3) if i != coord_idx]
    L1 = np.ptp(network["pore.coords"][:, other_axes[0]])
    L2 = np.ptp(network["pore.coords"][:, other_axes[1]])
    A = L1 * L2
    delta_C = 1.0
    D_eff = np.abs(diff_rate[0]) * L / (A * delta_C)
    print(f"Effective diffusivity: {D_eff:.6e} m^2/s")

    # Calculate tortuosity: tau = (D_bulk / D_eff) * porosity
    # Use the actual void fraction of the source image rather than Np/length^3.
    porosity = float(im.sum()) / im.size
    tau = D_bulk * porosity / D_eff
    print(f"Tortuosity: {tau:.4f}")

    # Save to VTK for ParaView
    proj = network.project
    op.io.project_to_vtk(project=proj, filename=os.path.join(output_dir, "fickian_diffusion"), fill_nans=0.0)
    print(f"Fickian diffusion results saved to '{output_dir}/fickian_diffusion.vtp'")

    return fd, D_eff, tau


def simulate_advection_diffusion(network, phase, axis="x"):
    """
    Simulate combined advection-diffusion to calculate concentration field,
    Peclet number, and combined flux.

    Parameters
    ----------
    axis : str
        The axis along which the flow and diffusion occur ('x', 'y', or 'z').

    Requires Stokes flow to be solved first (pressure field must exist).
    Results are saved as VTK for ParaView.
    """
    inlet_label, outlet_label, coord_idx = get_axis_faces(axis)

    # First solve Stokes flow to get the pressure field
    if "pore.pressure" not in phase.keys():
        print("Running Stokes flow first to establish velocity field...")
        simulate_stokes_flow(network, phase, axis=axis)

    # Add advection-diffusion conductance model
    phase.add_model(
        propname="throat.ad_dif_conductance",
        model=op.models.physics.ad_dif_conductance.ad_dif,
    )
    phase.regenerate_models()

    ad = op.algorithms.AdvectionDiffusion(network=network, phase=phase)

    inlet_pores = network.pores(inlet_label)
    outlet_pores = network.pores(outlet_label)

    ad.set_value_BC(pores=inlet_pores, values=1.0)
    ad.set_value_BC(pores=outlet_pores, values=0.0)
    ad.run()

    # Extract concentration field
    conc_ad = ad["pore.concentration"]
    phase["pore.concentration_ad"] = conc_ad


    # Calculate combined flux
    combined_rate = ad.rate(pores=outlet_pores, mode="group")
    print(f"Combined advection-diffusion rate ({axis}-axis): {combined_rate[0]:.6e}")

    # Calculate Peclet number: Pe = v * L / D
    D_bulk = phase["pore.diffusivity"].mean()
    L = np.ptp(network["pore.coords"][:, coord_idx])
    if "throat.velocity" in phase.keys():
        v_mean = phase["throat.velocity"].mean()
    else:
        v_mean = 1e-5  # fallback
    Pe = v_mean * L / D_bulk
    print(f"Mean Peclet number: {Pe:.4f}")

    # Save to VTK for ParaView
    proj = network.project
    op.io.project_to_vtk(project=proj, filename=os.path.join(output_dir, "advection_diffusion"), fill_nans=0.0)
    print(f"Advection-diffusion results saved to '{output_dir}/advection_diffusion.vtp'")

    return ad, Pe, combined_rate[0]


# Two Phase Flow Simulations


def simulate_drainage(network, invading_phase, defending_phase, axis="x"):
    """
    Simulate drainage (non-wetting phase displacing wetting phase).
    Plots and saves the capillary pressure-saturation curve.

    Parameters
    ----------
    axis : str
        The axis along which invasion occurs ('x', 'y', or 'z').

    Results are saved as VTK for ParaView.
    """
    inlet_label, outlet_label, coord_idx = get_axis_faces(axis)
    drain = op.algorithms.Drainage(network=network, phase=invading_phase)

    inlet_pores = network.pores(inlet_label)
    drain.set_inlet_BC(pores=inlet_pores)
    drain.run(pressures=50)

    # Get capillary pressure - saturation curve
    pc_data = drain.pc_curve()

    # Plot saturation curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pc_data.pc, pc_data.snwp, "b-o", markersize=4)
    ax.set_xlabel("Capillary Pressure [Pa]")
    ax.set_ylabel("Non-Wetting Phase Saturation")
    ax.set_title("Drainage: Capillary Pressure vs Saturation")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drainage_saturation_curve.png"), dpi=150)
    plt.show()
    print(f"Drainage saturation curve saved to '{output_dir}/drainage_saturation_curve.png'")

    # Store invasion data on phase for VTK export
    if "pore.invasion_pressure" in drain.keys():
        invading_phase["pore.invasion_pressure"] = drain["pore.invasion_pressure"]
    if "throat.invasion_pressure" in drain.keys():
        invading_phase["throat.invasion_pressure"] = drain["throat.invasion_pressure"]

    # Save to VTK for ParaView
    proj = network.project
    op.io.project_to_vtk(project=proj, filename=os.path.join(output_dir, "drainage"), fill_nans=0.0)
    print(f"Drainage results saved to '{output_dir}/drainage.vtp'")

    return drain, pc_data


def simulate_invasion_percolation(network, invading_phase, defending_phase, axis="x"):
    """
    Simulate invasion percolation (step-by-step invasion of non-wetting phase).
    Plots and saves the capillary pressure-saturation curve.

    Parameters
    ----------
    axis : str
        The axis along which invasion occurs ('x', 'y', or 'z').

    Results are saved as VTK for ParaView.
    """
    inlet_label, outlet_label, coord_idx = get_axis_faces(axis)
    ip = op.algorithms.InvasionPercolation(network=network, phase=invading_phase)

    inlet_pores = network.pores(inlet_label)
    ip.set_inlet_BC(pores=inlet_pores)
    ip.run()

    # Get capillary pressure - saturation curve
    pc_data = ip.pc_curve()

    # Plot saturation curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pc_data.pc, pc_data.snwp, "r-o", markersize=4)
    ax.set_xlabel("Capillary Pressure [Pa]")
    ax.set_ylabel("Non-Wetting Phase Saturation")
    ax.set_title("Invasion Percolation: Capillary Pressure vs Saturation")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "invasion_percolation_saturation_curve.png"), dpi=150)
    plt.show()
    print(f"IP saturation curve saved to '{output_dir}/invasion_percolation_saturation_curve.png'")

    # Store invasion data on phase for VTK export
    if "pore.invasion_pressure" in ip.keys():
        invading_phase["pore.invasion_pressure_ip"] = ip["pore.invasion_pressure"]
    if "throat.invasion_pressure" in ip.keys():
        invading_phase["throat.invasion_pressure_ip"] = ip["throat.invasion_pressure"]

    # Save to VTK for ParaView
    proj = network.project
    op.io.project_to_vtk(project=proj, filename=os.path.join(output_dir, "invasion_percolation"), fill_nans=0.0)
    print(f"Invasion percolation results saved to '{output_dir}/invasion_percolation.vtp'")

    return ip, pc_data


# ===================== MAIN EXECUTION =====================

# if __name__ == "__main__":
#     # Phase selection
#     print("Available phases: water, air, mercury")
#     phase_choice = input("Select a phase to add to the network: ")
#     phase = select_phase(pn, phase_choice)

#     # Axis selection
#     print("\nAvailable axes: x, y, z")
#     axis_choice = input("Select the axis along which simulations will run: ")

#     # --- Single Phase Flow Simulations ---
#     print("\n" + "=" * 50)
#     print("STOKES FLOW SIMULATION")
#     print("=" * 50)
#     sf, K = simulate_stokes_flow(pn, phase, axis=axis_choice)

#     print("\n" + "=" * 50)
#     print("FICKIAN DIFFUSION SIMULATION")
#     print("=" * 50)
#     fd, D_eff, tau = simulate_fickian_diffusion(pn, phase, axis=axis_choice)

#     print("\n" + "=" * 50)
#     print("ADVECTION-DIFFUSION SIMULATION")
#     print("=" * 50)
#     ad, Pe, combined_flux = simulate_advection_diffusion(pn, phase, axis=axis_choice)

#     # --- Two Phase Flow Simulations ---
#     print("\n" + "=" * 50)
#     print("TWO PHASE FLOW SIMULATIONS")
#     print("=" * 50)
#     print("Select a second phase for two-phase simulations.")
#     print("Available phases: water, air, mercury")
#     phase2_choice = input("Select the invading (non-wetting) phase: ")
#     invading_phase = select_phase(pn, phase2_choice)

#     print("\n--- Drainage Simulation ---")
#     drain, drain_pc = simulate_drainage(pn, invading_phase, phase, axis=axis_choice)

#     print("\n--- Invasion Percolation Simulation ---")
#     ip, ip_pc = simulate_invasion_percolation(pn, invading_phase, phase, axis=axis_choice)
