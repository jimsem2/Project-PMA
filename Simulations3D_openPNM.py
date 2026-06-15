from tokenize import Name
import matplotlib.pyplot as plt
import numpy as np
import porespy as ps
import imageio
import pandas as pd
import openpnm as op
import os

# Module-level state. These are populated by build_network() (called by the GUI
# or by the __main__ block below) instead of being built at import time, so that
# importing this module is cheap and the network can be built from any image.
im = None
pn = None
output_dir = "results_paraview"


def build_network(image, voxel_size=1):
    """Build an OpenPNM network from a PoreSpy binary image.

    Stores the image and network in module globals (im, pn) so the transport
    functions below can reference them as defaults, and returns the network.
    """
    global im, pn
    im = image
    os.makedirs(output_dir, exist_ok=True)

    im_snow2 = ps.networks.snow2(image, voxel_size=voxel_size)
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
    return pn

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

def select_phase(network=None, phase_name='water', diffusivity=None, viscosity=None,
                 contact_angle=None, surface_tension=None):  # returns phase
    # network: the pore network to which the phase will be added
    # phase_name: string indicating the type of phase ('water', 'air', 'mercury')
    # diffusivity / viscosity / contact_angle / surface_tension: optional overrides.
    #   When None, the code-defined default (or correlation model) for the phase is used.
    #   A unique phase name is generated automatically so the same network can host
    #   several phases across repeated simulation runs without name clashes.

    if network is None:
        network = pn
    phase_name = phase_name.strip().lower()
    if phase_name == "water":
        phase = op.phase.Water(network=network)
        # Add viscosity model for water (temperature-dependent) unless overridden.
        if viscosity is None:
            phase.add_model(propname='pore.viscosity',
                           model=op.models.phase.viscosity.water_correlation)
        else:
            phase["pore.viscosity"] = viscosity
        # OpenPNM's Water phase doesn't define diffusivity by default.
        # Use water self-diffusion coefficient at room temperature.
        phase["pore.diffusivity"] = 2.3e-9 if diffusivity is None else diffusivity  # m^2/s (25 °C)
        phase["pore.contact_angle"] = 110.0 if contact_angle is None else contact_angle  # degrees
    elif phase_name == "air":
        phase = op.phase.Air(network=network)
        # The Air phase already defines pore.viscosity via its default models,
        # so we only replace it when an explicit override is supplied.
        if viscosity is not None:
            phase["pore.viscosity"] = viscosity
        # Air phase lacks surface tension and contact angle. Use water-air
        # interfacial tension (~0.072 N/m) and assume air is the non-wetting
        # phase with contact angle 0° (measured through the invading phase).
        phase["pore.surface_tension"] = 0.072 if surface_tension is None else surface_tension  # N/m
        phase["pore.diffusivity"] = 2.5e-5 if diffusivity is None else diffusivity     # m^2/s (25 °C)
        phase["pore.contact_angle"] = 120.0 if contact_angle is None else contact_angle    # degrees
    elif phase_name == "mercury":
        phase = op.phase.Mercury(network=network)
        # Add constant viscosity for mercury (less sensitive to temperature)
        phase["pore.viscosity"] = 1.53e-3 if viscosity is None else viscosity  # Pa·s at room temp
        # Mercury phase doesn't define diffusivity by default.
        phase["pore.diffusivity"] = 1.7e-9 if diffusivity is None else diffusivity  # m^2/s (approx.)
        # Mercury intrusion: surface tension ~0.485 N/m, contact angle ~140°
        phase["pore.surface_tension"] = 0.485 if surface_tension is None else surface_tension
        phase["pore.contact_angle"] = 140.0 if contact_angle is None else contact_angle
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


def simulate_stokes_flow(network=None, chosen_phase='water', axis="x", inlet_pressure=1.0, outlet_pressure=0.0,
                         phase_overrides=None):
    # Simulate Stokes flow to calculate pressure gradients, velocity distributions, and Darcy permeability
    # axis: The axis along which the pressure drop is applied ('x', 'y', or 'z')
    # Results are saved as VTK for ParaView

    if network is None:
        network = pn
    phase = select_phase(network=network, phase_name=chosen_phase, **(phase_overrides or {}))

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
    A = op.topotools.get_domain_area(pn, inlets=inlet_pores, outlets=outlet_pores)  # Cross-sectional area from the other two axes
    L = op.topotools.get_domain_length(pn, inlets=inlet_pores, outlets=outlet_pores)    # length along chosen axis
    delta_P = inlet_pressure - outlet_pressure  # applied pressure difference
    K = np.abs(flow_rate[0]) * mu * L / (A * delta_P)
    print(f"Darcy permeability: {K:.6e} m^2")

    # Save to VTK for ParaView
    projSF = network.project
    op.io.project_to_vtk(project=projSF, filename=os.path.join(output_dir, "stokes_flow"), fill_nans=0.0)
    print(f"Stokes flow results saved to '{output_dir}/stokes_flow.vtp'")

    return sf, K


def simulate_fickian_diffusion(network=None, chosen_phase='water', axis="x", inlet_concentration=1.0, outlet_concentration=0.0,
                               phase_overrides=None):
    # Simulate Fickian diffusion to calculate concentration distributions, effective diffusivity, and tortuosity.
    # axis : The axis along which the concentration gradient is applied ('x', 'y', or 'z').
    # Results are saved as VTK for ParaView.

    if network is None:
        network = pn
    phase = select_phase(network=network, phase_name=chosen_phase, **(phase_overrides or {}))

    inlet_label, outlet_label, coord_idx = get_axis_faces(axis)
    fd = op.algorithms.FickianDiffusion(network=network, phase=phase)

    inlet_pores = network.pores(inlet_label)
    outlet_pores = network.pores(outlet_label)

    fd.set_value_BC(pores=inlet_pores, values=inlet_concentration)
    fd.set_value_BC(pores=outlet_pores, values=outlet_concentration)
    fd.run()

    # Extract concentration field
    concentration = fd["pore.concentration"]
    phase["pore.concentration"] = concentration

    # Calculate diffusive rate
    diff_rate = fd.rate(pores=outlet_pores, mode="group")
    print(f"Total diffusive rate through outlet ({axis}-axis): {diff_rate[0]:.6e}")

    # Calculate effective diffusivity: D_eff = J * L / (A * dC)
    D_bulk = phase["pore.diffusivity"].mean()
    A = op.topotools.get_domain_area(pn, inlets=inlet_pores, outlets=outlet_pores)  # Cross-sectional area from the other two axes
    L = op.topotools.get_domain_length(pn, inlets=inlet_pores, outlets=outlet_pores)    # length along chosen axis
    delta_C = inlet_concentration - outlet_concentration
    D_eff = np.abs(diff_rate[0]) * L / (A * delta_C)
    print(f"Effective diffusivity: {D_eff:.6e} m^2/s")

    # Calculate tortuosity: tau = (D_bulk / D_eff) * porosity
    # Use the actual void fraction of the source image rather than Np/length^3.
    porosity = float(im.sum()) / im.size
    tort = D_bulk * porosity / D_eff
    print(f"Tortuosity: {tort:.4f}")

    # Save to VTK for ParaView
    proj = network.project
    op.io.project_to_vtk(project=proj, filename=os.path.join(output_dir, "fickian_diffusion"), fill_nans=0.0)
    print(f"Fickian diffusion results saved to '{output_dir}/fickian_diffusion.vtp'")

    return fd, D_eff, tort


def simulate_advection_diffusion(network=None, chosen_phase='water', axis="x", inlet_pressure=1.0, outlet_pressure=0.0, inlet_concentration=1.0, outlet_concentration=0.0,
                                 phase_overrides=None):
    # Simulate combined advection-diffusion to calculate concentration field, Peclet number, and combined flux.
    # axis : The axis along which the flow and diffusion occur ('x', 'y', or 'z').
    # Requires Stokes flow to be solved first (pressure field must exist).
    # Results are saved as VTK for ParaView.

    if network is None:
        network = pn
    phase = select_phase(network=network, phase_name=chosen_phase, **(phase_overrides or {}))

    inlet_label, outlet_label, coord_idx = get_axis_faces(axis)

    # First solve Stokes flow on this same phase to establish the velocity field
    # (advection-diffusion needs throat.velocity / pore.pressure).
    if "pore.pressure" not in phase.keys():
        print("Running Stokes flow first to establish velocity field...")
        sf = op.algorithms.StokesFlow(network=network, phase=phase)
        sf.set_value_BC(pores=network.pores(inlet_label), values=inlet_pressure)
        sf.set_value_BC(pores=network.pores(outlet_label), values=outlet_pressure)
        sf.run()
        phase["pore.pressure"] = sf["pore.pressure"]
        throat_area = network["throat.cross_sectional_area"]
        throat_flow = sf.rate(throats=network.Ts, mode="single")
        phase["throat.velocity"] = np.abs(throat_flow / throat_area)

    # Add advection-diffusion conductance model
    phase.add_model(
        propname="throat.ad_dif_conductance",
        model=op.models.physics.ad_dif_conductance.ad_dif,
    )
    phase.regenerate_models()

    ad = op.algorithms.AdvectionDiffusion(network=network, phase=phase)

    inlet_pores = network.pores(inlet_label)
    outlet_pores = network.pores(outlet_label)

    ad.set_value_BC(pores=inlet_pores, values=inlet_concentration)
    ad.set_value_BC(pores=outlet_pores, values=outlet_concentration)
    ad.run()

    # Extract concentration field
    conc_ad = ad["pore.concentration"]
    phase["pore.concentration_ad"] = conc_ad


    # Calculate combined flux
    combined_rate = ad.rate(pores=outlet_pores, mode="group")
    print(f"Combined advection-diffusion rate ({axis}-axis): {combined_rate[0]:.6e}")

    # Calculate Peclet number: Pe = v * L / D
    D_bulk = phase["pore.diffusivity"].mean()
    L = op.topotools.get_domain_length(pn, inlets=inlet_pores, outlets=outlet_pores)    # length along chosen axis
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


def simulate_drainage(network=None, inv_phase='air', def_phase='water', axis="x", save_excel=False,
                      phase_overrides=None):
    # Simulate drainage (non-wetting phase displacing wetting phase).
    # Plots and saves the capillary pressure-saturation curve.
    # axis : The axis along which invasion occurs ('x', 'y', or 'z').
    # Results are saved as VTK for ParaView.
    # save_excel: if True, saves the saturation curve data in an Excel file, default False

    if network is None:
        network = pn
    invading_phase = select_phase(network=network, phase_name=inv_phase, **(phase_overrides or {}))
    defending_phase = select_phase(network=network, phase_name=def_phase)

    inlet_label, outlet_label, coord_idx = get_axis_faces(axis)
    drain = op.algorithms.Drainage(network=network, phase=invading_phase)

    inlet_pores = network.pores(inlet_label)
    drain.set_inlet_BC(pores=inlet_pores)
    drain.run(pressures=50)         # Run drainage with 50 pressure steps (default is 25). Adjust as needed for resolution.

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

    # Save saturation curve data to Excel if requested
    if save_excel:
        pd.DataFrame({'Capillary Pressure [Pa]': pc_data.pc, 'Non-Wetting Phase Saturation': pc_data.snwp}).to_excel('Drainage_openPNM_Saturation_Curve.xlsx', index=False)
        print("Plot saved to Drainage_openPNM_Saturation_Curve.xlsx")

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


def simulate_invasion_percolation(network=None, inv_phase='air', def_phase='water', axis="x", save_excel=False,
                                  phase_overrides=None):
    # Simulate invasion percolation (step-by-step invasion of non-wetting phase).
    # Plots and saves the capillary pressure-saturation curve.
    # axis : The axis along which invasion occurs ('x', 'y', or 'z').
    # Results are saved as VTK for ParaView.
    # save_excel: if True, saves the saturation curve data in an Excel file, default False

    if network is None:
        network = pn
    invading_phase = select_phase(network=network, phase_name=inv_phase, **(phase_overrides or {}))
    defending_phase = select_phase(network=network, phase_name=def_phase)

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

    # Save saturation curve data to Excel if requested
    if save_excel:
        pd.DataFrame({'Capillary Pressure [Pa]': pc_data.pc, 'Non-Wetting Phase Saturation': pc_data.snwp}).to_excel('InvasionPercolation_openPNM_Saturation_Curve.xlsx', index=False)
        print("Plot saved to InvasionPercolation_openPNM_Saturation_Curve.xlsx")

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

if __name__ == "__main__":
    # Build a default network from a generated blob image when run standalone.
    length = 60
    build_network(ps.generators.blobs(shape=[length, length, length], porosity=0.6, blobiness=2, seed=10))

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
