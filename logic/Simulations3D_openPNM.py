import matplotlib.pyplot as plt
import numpy as np
import porespy as ps
import pandas as pd
import openpnm as op

# Show minor tick marks on all charts produced by this module
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True

# Module-level state. Populated by build_network() (called by the GUI)
# instead of being built at import time, so importing this module is cheap
# and the network can be built from any image.
im = None
pn = None
_VOXEL_SIZE = None   # metres per voxel of the current image; set by build_network
def build_network(image, voxel_size=1):
    """Build an OpenPNM network from a PoreSpy binary image.

    Stores the image and network in module globals (im, pn) so the transport
    functions below can reference them as defaults, and returns the network.
    The voxel size is also stored (``_VOXEL_SIZE``) so the transport functions
    can recover the true physical dimensions of the sample cuboid (image shape
    x voxel size) for the recommended, image-based normalisation of K / D_eff.
    """
    global im, pn, _VOXEL_SIZE
    im = image
    _VOXEL_SIZE = voxel_size

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


def image_domain_dimensions(coord_idx):
    """Physical cross-sectional area A (m^2) and length L (m) of the sample,
    taken from the *true* image cuboid: L = N_axis * voxel_size and
    A = product of the two perpendicular (N * voxel_size) extents.

    This is the physically correct normalisation for Darcy permeability and
    effective diffusivity — it is exactly the macroscopic sample cross-section
    a laboratory permeameter uses, and the image is a clean rectangular block
    of known size (unlike the irregular extracted network). Returns (A, L), or
    (None, None) if the image / voxel size are unknown (e.g. a network was
    passed in without going through build_network).
    """
    if im is None or not _VOXEL_SIZE:
        return None, None
    perp = [i for i in range(3) if i != coord_idx]
    L = im.shape[coord_idx] * _VOXEL_SIZE
    A = float(np.prod([im.shape[p] * _VOXEL_SIZE for p in perp]))
    return A, L

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
        phase["pore.diffusivity"] = 2.5e-5 if diffusivity is None else diffusivity     # m^2/s (25 °C)
        phase["pore.contact_angle"] = 120.0 if contact_angle is None else contact_angle    # degrees
        phase["pore.surface_tension"] = 0.072 if surface_tension is None else surface_tension  # N/m
    elif phase_name == "mercury":
        phase = op.phase.Mercury(network=network)
        # Add constant viscosity for mercury (less sensitive to temperature)
        phase["pore.viscosity"] = 1.53e-3 if viscosity is None else viscosity  # Pa·s at room temp
        # Mercury phase doesn't define diffusivity by default.
        phase["pore.diffusivity"] = 1.7e-9 if diffusivity is None else diffusivity  # m^2/s (approx.)
        # Mercury intrusion: surface tension ~0.485 N/m, contact angle ~140°
        phase["pore.contact_angle"] = 140.0 if contact_angle is None else contact_angle
        phase["pore.surface_tension"] = 0.485 if surface_tension is None else surface_tension
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


def export_vtk(network=None, filename="openpnm_results"):
    """Export the whole project (network + every phase on it) to one VTK PolyData file.

    ``filename`` is a path stem *without* extension — OpenPNM appends ``.vtp``.
    Because a single OpenPNM project holds all the phases created across the
    simulations run in a session, this one file contains every result field
    (pressure, concentration, invasion pressures, ...) in a single ParaView load.
    """
    if network is None:
        network = pn
    op.io.project_to_vtk(project=network.project, filename=filename, fill_nans=0.0)
    print(f"OpenPNM results saved to '{filename}.vtp'")
    return f"{filename}.vtp"


def simulate_stokes_flow(network=None, chosen_phase='water', axis="x", inlet_pressure=1.0, outlet_pressure=0.0,
                         phase_overrides=None, save_vtk=True):
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
    # Note: negative sign is correct — OpenPNM convention: positive = flow INTO a pore.
    # Outlet pores have fluid flowing OUT → negative. Magnitude = volumetric flow rate.
    flow_rate = sf.rate(pores=outlet_pores, mode="group")
    Q = np.abs(flow_rate[0])
    print(f"Total flow rate through outlet ({axis}-axis): {flow_rate[0]:.5e} m³/s  "
          f"(negative = fluid exiting domain, magnitude = {Q:.5e} m³/s)")

    # Calculate velocity at throats (Q / A)
    throat_area = network["throat.cross_sectional_area"]
    throat_flow = sf.rate(throats=network.Ts, mode="single")
    velocity = np.abs(throat_flow / throat_area)
    phase["throat.velocity"] = velocity

    # Interpolate throat velocity to pore centres (average of adjacent throats)
    # so that pore.velocity appears as point data in ParaView and can colour Glyph spheres
    conns = network["throat.conns"]
    pore_vel = np.zeros(network.Np)
    counts = np.zeros(network.Np, dtype=int)
    np.add.at(pore_vel, conns[:, 0], velocity)
    np.add.at(pore_vel, conns[:, 1], velocity)
    np.add.at(counts, conns[:, 0], 1)
    np.add.at(counts, conns[:, 1], 1)
    pore_vel = np.where(counts > 0, pore_vel / counts, 0.0)
    phase["pore.velocity"] = pore_vel

    # ---- Darcy permeability: K = Q * mu * L / (A * dP) --------------------
    # The Stokes problem is solved ONCE: Q (flow rate) and delta_P are the same
    # for every value below. The only thing that changes K is how the domain
    # cross-section A and length L are measured — a genuinely ambiguous choice
    # for an irregular, image-extracted network with no flat faces. Three
    # estimates are reported; they can differ by tens of percent (or more for
    # thin / coarse networks). See the note printed after them.
    mu = phase["pore.viscosity"].mean()
    delta_P = inlet_pressure - outlet_pressure
    _D  = 9.869233e-13   # 1 Darcy in m²
    _mD = 9.869233e-16   # 1 mD in m²
    _ax = coord_idx
    _perp = [i for i in range(3) if i != _ax]
    coords = pn["pore.coords"]

    def _report_K(label, A, L):
        K_val = Q * mu * L / (A * delta_P)
        print(f"  Darcy permeability [{label}]:")
        print(f"    A = {A:.5e} m²   L = {L:.5e} m")
        print(f"    K = {K_val:.5e} m²  |  {K_val/_D:.5e} D  |  {K_val/_mD:.5e} mD")
        return K_val

    # (1) Image-based — RECOMMENDED. True sample cuboid dimensions from the
    #     image (N_voxels * voxel_size); matches how a lab permeameter defines K.
    A_img, L_img = image_domain_dimensions(_ax)
    K_img = _report_K("image-based — true sample dimensions (RECOMMENDED)",
                      A_img, L_img) if A_img is not None else None

    # (2) Coordinate-based — robust approximation. Bounding box of pore
    #     centroids; systematically off by ~one pore radius (centroids do not
    #     reach the true image faces) but never fails.
    L_coord = abs(coords[outlet_pores, _ax].mean() - coords[inlet_pores, _ax].mean())
    A_coord = float(np.prod([(coords[:, p].max() - coords[:, p].min()) for p in _perp]))
    K_coord = _report_K("coordinate-based — pore-centroid bounding box", A_coord, L_coord)

    # (3) Geometric — OpenPNM get_domain_area/length (convex hull of face pores).
    #     Physically motivated but fragile: it under-covers the true face and
    #     can be badly wrong for sparse networks. Use only as a sanity check.
    try:
        A_geo = op.topotools.get_domain_area(pn, inlets=inlet_pores, outlets=outlet_pores)
        L_geo = op.topotools.get_domain_length(pn, inlets=inlet_pores, outlets=outlet_pores)
        _report_K("geometric — OpenPNM get_domain_area/length (sanity check only)",
                  A_geo, L_geo)
    except Exception as _e:
        print(f"  Darcy permeability [geometric method]: could not compute ({_e})")

    # Primary value: image-based when available, else the robust coordinate one.
    K = K_img if K_img is not None else K_coord
    print(f"  -> Reported permeability K = {K:.5e} m² "
          f"({'image-based' if K_img is not None else 'coordinate-based'}).")

    # Save to VTK for ParaView (skipped when the caller exports the whole project once).
    if save_vtk:
        projSF = network.project
        op.io.project_to_vtk(project=projSF, filename="openpnm_stokes_flow", fill_nans=0.0)
        print("Stokes flow results saved to 'openpnm_stokes_flow.vtp'")

    return sf, K


def simulate_fickian_diffusion(network=None, chosen_phase='water', axis="x", inlet_concentration=1.0, outlet_concentration=0.0,
                               phase_overrides=None, save_vtk=True):
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

    # Interpolate pore concentration to throat centres so tubes can be coloured by concentration
    _conns = network["throat.conns"]
    phase["throat.concentration"] = (concentration[_conns[:, 0]] + concentration[_conns[:, 1]]) / 2

    # Diffusive rate — negative = flux exiting domain at outlet (OpenPNM sign convention).
    # Unit: [concentration-unit] × m³/s.
    #   → mol/s  if inlet_concentration / outlet_concentration are in mol/m³
    #   → m³/s   if they are dimensionless (0–1); D_eff and tortuosity are unaffected either way.
    diff_rate = fd.rate(pores=outlet_pores, mode="group")
    J = np.abs(diff_rate[0])
    c_unit = "mol/s (if C in mol/m³) or m³/s (if C dimensionless)"
    print(f"Total diffusive rate through outlet ({axis}-axis): {diff_rate[0]:.5e}  [{c_unit}]  "
          f"(negative = flux exiting domain, magnitude = {J:.5e})")

    # ---- Effective diffusivity: D_eff = J * L / (A * dC) ------------------
    # As with permeability, the diffusion problem is solved ONCE (J and delta_C
    # are fixed); the three values below differ only in how the domain
    # cross-section A and length L are measured. See the note after them.
    D_bulk = phase["pore.diffusivity"].mean()
    delta_C = inlet_concentration - outlet_concentration
    _ax = coord_idx
    _perp = [i for i in range(3) if i != _ax]
    _coords = pn["pore.coords"]

    def _report_D(label, A, L):
        D_val = J * L / (A * delta_C)
        print(f"  Effective diffusivity [{label}]:")
        print(f"    A = {A:.5e} m²   L = {L:.5e} m   D_eff = {D_val:.5e} m²/s")
        return D_val

    # (1) Image-based — RECOMMENDED (true sample cuboid dimensions).
    A_img, L_img = image_domain_dimensions(_ax)
    D_eff_img = _report_D("image-based — true sample dimensions (RECOMMENDED)",
                          A_img, L_img) if A_img is not None else None

    # (2) Coordinate-based — robust approximation (pore-centroid bounding box).
    L_coord = abs(_coords[outlet_pores, _ax].mean() - _coords[inlet_pores, _ax].mean())
    A_coord = float(np.prod([(_coords[:, p].max() - _coords[:, p].min()) for p in _perp]))
    D_eff_coord = _report_D("coordinate-based — pore-centroid bounding box", A_coord, L_coord)

    # (3) Geometric — OpenPNM get_domain_area/length (sanity check only).
    try:
        A_geo = op.topotools.get_domain_area(pn, inlets=inlet_pores, outlets=outlet_pores)
        L_geo = op.topotools.get_domain_length(pn, inlets=inlet_pores, outlets=outlet_pores)
        D_eff_geo = _report_D("geometric — OpenPNM get_domain_area/length (sanity check only)",
                              A_geo, L_geo)
    except Exception as _e:
        print(f"  Effective diffusivity [geometric method]: could not compute ({_e})")
        D_eff_geo = None

    # Primary value: image-based when available, else the robust coordinate one.
    D_eff = D_eff_img if D_eff_img is not None else D_eff_coord
    print(f"  -> Reported effective diffusivity D_eff = {D_eff:.5e} m²/s "
          f"({'image-based' if D_eff_img is not None else 'coordinate-based'}).")

    # ---- Tortuosity ------------------------------------------------------
    # NOTE — this is the NETWORK diffusive tortuosity FACTOR, defined through
    # the effective-medium relation  D_eff = (porosity / tau) * D_bulk, i.e.
    #     tau = porosity * D_bulk / D_eff .
    # It measures how much the pore network lengthens/constricts diffusion paths
    # relative to open bulk, on the *extracted pore network*. It is a DIFFERENT
    # quantity, computed by a DIFFERENT method, from the voxel tortuosity of
    # Simulations3D_porespy.TORTUOSITY (which solves finite-difference Laplace
    # diffusion on the voxel image directly via ps.simulations.tortuosity_fd).
    # The two normally disagree — the network idealises the geometry, the voxel
    # method keeps every voxel — and comparing them is a useful sanity check,
    # not a contradiction. Because tau is inversely proportional to D_eff, it
    # inherits the A/L ambiguity above, so it is reported for the image-based
    # (physical) normalisation and, for transparency, for the geometric one.
    porosity = float(im.sum()) / im.size
    tort = D_bulk * porosity / D_eff       # from the reported (image-based) D_eff
    print(f"Network diffusive tortuosity factor tau = {tort:.5f}  "
          f"(porosity*D_bulk/D_eff, image-based normalisation; "
          f"distinct from the voxel-based TORTUOSITY / tortuosity_fd metric)")
    if D_eff_geo is not None:
        print(f"    (for comparison, geometric-normalisation tau = "
              f"{D_bulk * porosity / D_eff_geo:.5f})")

    # Save to VTK for ParaView (skipped when the caller exports the whole project once).
    if save_vtk:
        proj = network.project
        op.io.project_to_vtk(project=proj, filename="openpnm_fickian_diffusion", fill_nans=0.0)
        print("Fickian diffusion results saved to 'openpnm_fickian_diffusion.vtp'")

    return fd, D_eff, tort


def simulate_advection_diffusion(network=None, chosen_phase='water', axis="x", inlet_pressure=1.0, outlet_pressure=0.0, inlet_concentration=1.0, outlet_concentration=0.0,
                                 phase_overrides=None, save_vtk=True):
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
    # NOTE — test for a *solved* field, not for the key: every OpenPNM phase is
    # created with a uniform default pore.pressure = 101325 Pa, so a plain
    # `"pore.pressure" not in phase.keys()` is never true. With that test the
    # pre-solve was skipped, the field stayed uniform, every throat flow was
    # zero, and the advection-diffusion result silently collapsed onto the pure
    # Fickian one.
    if np.ptp(phase["pore.pressure"]) == 0:
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

    # Interpolate pore concentration to throat centres for tube coloring in ParaView
    _conns_ad = network["throat.conns"]
    phase["throat.concentration_ad"] = (conc_ad[_conns_ad[:, 0]] + conc_ad[_conns_ad[:, 1]]) / 2

    # The advection-diffusion conductance is a multi-column (directional) array, so
    # project_to_vtk can't export it and it never appears in ParaView. Store a scalar
    # per-throat *representative* value (mean over the directional columns) so it can
    # be used to colour the throat tubes. Note: this is only a summary — the true
    # conductance is direction-dependent (upwinded by the local flow).
    _g_ad = phase["throat.ad_dif_conductance"]
    phase["throat.ad_dif_conductance_scalar"] = _g_ad if _g_ad.ndim == 1 else _g_ad.mean(axis=1)

    # Combined rate — negative = total flux exiting domain (OpenPNM sign convention)
    combined_rate = ad.rate(pores=outlet_pores, mode="group")
    J_ad = np.abs(combined_rate[0])
    print(f"Combined advection-diffusion rate ({axis}-axis): {combined_rate[0]:.5e} m³/s  "
          f"(negative = flux exiting domain, magnitude = {J_ad:.5e} m³/s)")

    # ---- Peclet number ---------------------------------------------------
    # Pe = v L / D is a *local* quantity: the length that matters is the one
    # over which advection competes with diffusion, i.e. the throat length, not
    # the sample length. Taken per throat it is exactly the ratio the upwinding
    # scheme itself uses, Pe_ij = Q_ij / g_d,ij, because the diffusive
    # conductance already carries the throat geometry (g_d = D A / L_throat, so
    # Q/g_d = v L_throat / D). The reported value is the average over throats.
    #
    # A domain-scale Pe (same velocity, but with the whole sample as L) is also
    # printed for context; it is larger by roughly L_sample / L_throat and says
    # nothing about whether advection alters the pore-scale solve.
    P_ad = phase["pore.pressure"]
    cn_pe = network["throat.conns"]
    gh_pe = phase["throat.hydraulic_conductance"]
    gd_pe = phase["throat.diffusive_conductance"]
    if np.ndim(gd_pe) > 1:
        gd_pe = np.mean(gd_pe, axis=1)
    Qij_pe = np.abs(gh_pe * (P_ad[cn_pe[:, 0]] - P_ad[cn_pe[:, 1]]))
    Pe_throats = Qij_pe / gd_pe
    Pe = float(np.mean(Pe_throats))
    print(f"Mean throat Peclet number: {Pe:.5f}  "
          f"(median {np.median(Pe_throats):.5f}, max {Pe_throats.max():.5f}; "
          f"Pe > 1 means advection dominates within a throat)")

    D_bulk = phase["pore.diffusivity"].mean()
    _A_img, L = image_domain_dimensions(coord_idx)
    if L is None:
        _ax_ad = coord_idx
        _c_ad = pn["pore.coords"]
        try:
            L = abs(_c_ad[outlet_pores, _ax_ad].mean() - _c_ad[inlet_pores, _ax_ad].mean())
        except Exception:
            L = op.topotools.get_domain_length(pn, inlets=inlet_pores, outlets=outlet_pores)
    if "throat.velocity" not in phase.keys():
        raise RuntimeError("throat.velocity missing — the Stokes pre-solve did not run")
    v_mean = phase["throat.velocity"].mean()
    Pe_domain = v_mean * L / D_bulk
    print(f"  domain-scale Peclet (v_mean * sample length / D): {Pe_domain:.5f}  "
          f"(mean throat velocity {v_mean:.5e} m/s over L = {L:.5e} m)")

    # Save to VTK for ParaView (skipped when the caller exports the whole project once).
    if save_vtk:
        proj = network.project
        op.io.project_to_vtk(project=proj, filename="openpnm_advection_diffusion", fill_nans=0.0)
        print("Advection-diffusion results saved to 'openpnm_advection_diffusion.vtp'")

    return ad, Pe, combined_rate[0]


# Two Phase Flow Simulations


def simulate_drainage(network=None, inv_phase='air', def_phase='water', axis="x", save_excel=False,
                      phase_overrides=None, save_vtk=True):
    # Simulate drainage (non-wetting phase displacing wetting phase).
    # Plots and saves the capillary pressure-saturation curve.
    # axis : The axis along which invasion occurs ('x', 'y', or 'z').
    # Results are saved as VTK for ParaView.
    # save_excel: if True, saves the saturation curve data in an Excel file, default False

    if network is None:
        network = pn
    invading_phase = select_phase(network=network, phase_name=inv_phase, **(phase_overrides or {}))
    defending_phase = select_phase(network=network, phase_name=def_phase)

    drain = op.algorithms.Drainage(network=network, phase=invading_phase)

    if axis == "all":
        inlet_pores = np.unique(np.concatenate([
            network.pores(face)
            for face in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
            if network.pores(face).size > 0
        ]))
    else:
        inlet_label, outlet_label, coord_idx = get_axis_faces(axis)
        inlet_pores = network.pores(inlet_label)
    drain.set_inlet_BC(pores=inlet_pores)
    # Run at every unique throat entry pressure (maximum resolution).
    # Passing pressures=N would sample only N uniformly-spaced levels, causing
    # fewer curve points than the actual number of distinct throat radii.
    drain.run()

    # Get capillary pressure - saturation curve
    pc_data = drain.pc_curve()

    # Plot saturation curve
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.canvas.manager.set_window_title("OpenPNM: Drainage - Saturation Curve")
    ax.step(pc_data.pc, pc_data.snwp, "b-", where="post", linewidth=1.2)
    ax.scatter(pc_data.pc, pc_data.snwp, s=20, color="blue", zorder=3)
    ax.set_xlabel("Capillary Pressure [Pa]")
    ax.set_ylabel("Non-Wetting Phase Saturation")
    ax.set_title("Drainage: Capillary Pressure vs Saturation")
    plt.tight_layout()
    plt.show(block=False)

    # Save saturation curve data to Excel if requested
    if save_excel:
        pd.DataFrame({'Capillary Pressure [Pa]': pc_data.pc, 'Non-Wetting Phase Saturation': pc_data.snwp}).to_excel('Drainage_openPNM_Saturation_Curve.xlsx', index=False)
        print("Plot saved to Drainage_openPNM_Saturation_Curve.xlsx")

    # Store invasion data on phase for VTK export
    if "pore.invasion_pressure" in drain.keys():
        invading_phase["pore.invasion_pressure"] = drain["pore.invasion_pressure"]
    if "throat.invasion_pressure" in drain.keys():
        invading_phase["throat.invasion_pressure"] = drain["throat.invasion_pressure"]

    # Save to VTK for ParaView (skipped when the caller exports the whole project once).
    if save_vtk:
        proj = network.project
        op.io.project_to_vtk(project=proj, filename="openpnm_drainage", fill_nans=0.0)
        print("Drainage results saved to 'openpnm_drainage.vtp'")

    return drain, pc_data


def simulate_invasion_percolation(network=None, inv_phase='air', def_phase='water', axis="x", save_excel=False,
                                  phase_overrides=None, save_vtk=True):
    # Simulate invasion percolation (step-by-step invasion of non-wetting phase).
    # Plots and saves the capillary pressure-saturation curve.
    # axis : The axis along which invasion occurs ('x', 'y', or 'z').
    # Results are saved as VTK for ParaView.
    # save_excel: if True, saves the saturation curve data in an Excel file, default False

    if network is None:
        network = pn
    invading_phase = select_phase(network=network, phase_name=inv_phase, **(phase_overrides or {}))
    defending_phase = select_phase(network=network, phase_name=def_phase)

    ip = op.algorithms.InvasionPercolation(network=network, phase=invading_phase)

    if axis == "all":
        inlet_pores = np.unique(np.concatenate([
            network.pores(face)
            for face in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
            if network.pores(face).size > 0
        ]))
    else:
        inlet_label, outlet_label, coord_idx = get_axis_faces(axis)
        inlet_pores = network.pores(inlet_label)
    ip.set_inlet_BC(pores=inlet_pores)
    ip.run()

    # Get capillary pressure - saturation curve
    pc_data = ip.pc_curve()

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.canvas.manager.set_window_title("OpenPNM: Invasion Percolation - Saturation Curve")
    ax.plot(pc_data.pc, pc_data.snwp, "r-o", markersize=4)
    ax.set_xlabel("Capillary Pressure [Pa]")
    ax.set_ylabel("Non-Wetting Phase Saturation")
    ax.set_title("Invasion Percolation: Capillary Pressure vs Saturation")
    plt.tight_layout()
    plt.show(block=False)

    # Save saturation curve data to Excel if requested
    if save_excel:
        pd.DataFrame({'Capillary Pressure [Pa]': pc_data.pc, 'Non-Wetting Phase Saturation': pc_data.snwp}).to_excel('InvasionPercolation_openPNM_Saturation_Curve.xlsx', index=False)
        print("Plot saved to InvasionPercolation_openPNM_Saturation_Curve.xlsx")

    # Store invasion data on phase for VTK export
    if "pore.invasion_pressure" in ip.keys():
        invading_phase["pore.invasion_pressure_ip"] = ip["pore.invasion_pressure"]
    if "throat.invasion_pressure" in ip.keys():
        invading_phase["throat.invasion_pressure_ip"] = ip["throat.invasion_pressure"]

    # Save to VTK for ParaView (skipped when the caller exports the whole project once).
    if save_vtk:
        proj = network.project
        op.io.project_to_vtk(project=proj, filename="openpnm_invasion_percolation", fill_nans=0.0)
        print("Invasion percolation results saved to 'openpnm_invasion_percolation.vtp'")

    return ip, pc_data



