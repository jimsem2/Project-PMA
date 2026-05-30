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

output_dir = "results_paraview"
os.makedirs(output_dir, exist_ok=True)


def select_phase(network, phase_name):
    """
    Create and add a phase to the network.

    Parameters
    ----------
    network : op.network.Network
        The pore network to add the phase to.
    phase_name : str
        Name of the phase: 'water', 'air', or 'mercury'.

    Returns
    -------
    phase : op.phase object
    """
    phase_name = phase_name.strip().lower()
    if phase_name == "water":
        phase = op.phase.Water(network=network, name="water")
    elif phase_name == "air":
        phase = op.phase.Air(network=network, name="air")
    elif phase_name == "mercury":
        phase = op.phase.Mercury(network=network, name="mercury")
    else:
        raise ValueError(f"Unknown phase '{phase_name}'. Choose 'water', 'air', or 'mercury'.")

    phase.add_model(
        propname="throat.hydraulic_conductance",
        model=op.models.physics.hydraulic_conductance.hagen_poiseuille,
    )
    phase.add_model(
        propname="throat.diffusive_conductance",
        model=op.models.physics.diffusive_conductance.ordinary_diffusion,
    )
    phase.add_model(
        propname="throat.entry_pressure",
        model=op.models.physics.capillary_pressure.washburn,
    )
    phase.regenerate_models()
    print(f"Phase '{phase_name}' added to the network.")
    return phase


# Single Phase Flow Simulations


def simulate_stokes_flow(network, phase):
    """
    Simulate Stokes flow to calculate pressure gradients, velocity
    distributions, and Darcy permeability.

    Results are saved as VTK for ParaView.
    """
    sf = op.algorithms.StokesFlow(network=network, phase=phase)

    # Boundary conditions: pressure drop along x-axis
    inlet_pores = network.pores("left")
    outlet_pores = network.pores("right")

    sf.set_value_BC(pores=inlet_pores, values=1.0)
    sf.set_value_BC(pores=outlet_pores, values=0.0)
    sf.run()

    # Extract pressure field
    pressure = sf["pore.pressure"]
    phase["pore.pressure"] = pressure

    # Calculate flow rate through the outlet
    flow_rate = sf.rate(pores=outlet_pores, mode="group")
    print(f"Total flow rate through outlet: {flow_rate[0]:.6e}")

    # Calculate velocity at throats (Q / A)
    throat_area = network["throat.cross_sectional_area"]
    throat_flow = sf.rate(throats=network.Ts, mode="single")
    velocity = np.abs(throat_flow) / throat_area
    phase["throat.velocity"] = velocity

    # Calculate Darcy permeability: K = Q * mu * L / (A * dP)
    mu = phase["pore.viscosity"].mean()
    L = np.ptp(network["pore.coords"][:, 0])  # length along x
    A = L * L  # cross-sectional area (assuming cubic domain)
    delta_P = 1.0  # applied pressure difference
    K = np.abs(flow_rate[0]) * mu * L / (A * delta_P)
    print(f"Darcy permeability: {K:.6e} m^2")

    # Save to VTK for ParaView
    proj = network.project
    op.io.project_to_vtk(project=proj, filename=os.path.join(output_dir, "stokes_flow"), fill_nans=0.0)
    print(f"Stokes flow results saved to '{output_dir}/stokes_flow.vtp'")

    return sf, K


def simulate_fickian_diffusion(network, phase):
    """
    Simulate Fickian diffusion to calculate concentration distributions,
    effective diffusivity, and tortuosity.

    Results are saved as VTK for ParaView.
    """
    fd = op.algorithms.FickianDiffusion(network=network, phase=phase)

    inlet_pores = network.pores("left")
    outlet_pores = network.pores("right")

    fd.set_value_BC(pores=inlet_pores, values=1.0)
    fd.set_value_BC(pores=outlet_pores, values=0.0)
    fd.run()

    # Extract concentration field
    concentration = fd["pore.concentration"]
    phase["pore.concentration"] = concentration

    # Calculate diffusive rate
    diff_rate = fd.rate(pores=outlet_pores, mode="group")
    print(f"Total diffusive rate through outlet: {diff_rate[0]:.6e}")

    # Calculate effective diffusivity: D_eff = J * L / (A * dC)
    D_bulk = phase["pore.diffusivity"].mean()
    L = np.ptp(network["pore.coords"][:, 0])
    A = L * L
    delta_C = 1.0
    D_eff = np.abs(diff_rate[0]) * L / (A * delta_C)
    print(f"Effective diffusivity: {D_eff:.6e} m^2/s")

    # Calculate tortuosity: tau = D_bulk / D_eff * porosity
    porosity = network.Np / (length ** 3)  # approximate porosity
    tau = D_bulk * porosity / D_eff
    print(f"Tortuosity: {tau:.4f}")

    # Save to VTK for ParaView
    proj = network.project
    op.io.project_to_vtk(project=proj, filename=os.path.join(output_dir, "fickian_diffusion"), fill_nans=0.0)
    print(f"Fickian diffusion results saved to '{output_dir}/fickian_diffusion.vtp'")

    return fd, D_eff, tau


def simulate_advection_diffusion(network, phase):
    """
    Simulate combined advection-diffusion to calculate concentration field,
    Peclet number, and combined flux.

    Requires Stokes flow to be solved first (pressure field must exist).
    Results are saved as VTK for ParaView.
    """
    # First solve Stokes flow to get the pressure field
    if "pore.pressure" not in phase.keys():
        print("Running Stokes flow first to establish velocity field...")
        simulate_stokes_flow(network, phase)

    # Add advection-diffusion conductance model
    phase.add_model(
        propname="throat.ad_dif_conductance",
        model=op.models.physics.ad_dif_conductance.ad_dif,
    )
    phase.regenerate_models()

    ad = op.algorithms.AdvectionDiffusion(network=network, phase=phase)

    inlet_pores = network.pores("left")
    outlet_pores = network.pores("right")

    ad.set_value_BC(pores=inlet_pores, values=1.0)
    ad.set_value_BC(pores=outlet_pores, values=0.0)
    ad.run()

    # Extract concentration field
    conc_ad = ad["pore.concentration"]
    phase["pore.concentration_ad"] = conc_ad

    # Calculate combined flux
    combined_rate = ad.rate(pores=outlet_pores, mode="group")
    print(f"Combined advection-diffusion rate: {combined_rate[0]:.6e}")

    # Calculate Peclet number: Pe = v * L / D
    D_bulk = phase["pore.diffusivity"].mean()
    L = np.ptp(network["pore.coords"][:, 0])
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


def simulate_drainage(network, invading_phase, defending_phase):
    """
    Simulate drainage (non-wetting phase displacing wetting phase).
    Plots and saves the capillary pressure-saturation curve.

    Results are saved as VTK for ParaView.
    """
    drain = op.algorithms.Drainage(phase=invading_phase)

    inlet_pores = network.pores("left")
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


def simulate_invasion_percolation(network, invading_phase, defending_phase):
    """
    Simulate invasion percolation (step-by-step invasion of non-wetting phase).
    Plots and saves the capillary pressure-saturation curve.

    Results are saved as VTK for ParaView.
    """
    ip = op.algorithms.InvasionPercolation(phase=invading_phase)

    inlet_pores = network.pores("left")
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

if __name__ == "__main__":
    # Phase selection
    print("Available phases: water, air, mercury")
    phase_choice = input("Select a phase to add to the network: ")
    phase = select_phase(pn, phase_choice)

    # --- Single Phase Flow Simulations ---
    print("\n" + "=" * 50)
    print("STOKES FLOW SIMULATION")
    print("=" * 50)
    sf, K = simulate_stokes_flow(pn, phase)

    print("\n" + "=" * 50)
    print("FICKIAN DIFFUSION SIMULATION")
    print("=" * 50)
    fd, D_eff, tau = simulate_fickian_diffusion(pn, phase)

    print("\n" + "=" * 50)
    print("ADVECTION-DIFFUSION SIMULATION")
    print("=" * 50)
    ad, Pe, combined_flux = simulate_advection_diffusion(pn, phase)

    # --- Two Phase Flow Simulations ---
    print("\n" + "=" * 50)
    print("TWO PHASE FLOW SIMULATIONS")
    print("=" * 50)
    print("Select a second phase for two-phase simulations.")
    print("Available phases: water, air, mercury")
    phase2_choice = input("Select the invading (non-wetting) phase: ")
    invading_phase = select_phase(pn, phase2_choice)

    print("\n--- Drainage Simulation ---")
    drain, drain_pc = simulate_drainage(pn, invading_phase, phase)

    print("\n--- Invasion Percolation Simulation ---")
    ip, ip_pc = simulate_invasion_percolation(pn, invading_phase, phase)
