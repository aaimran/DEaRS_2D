#!/usr/bin/env python3
"""
2D Elastic Wave Equation Earthquake Rupture Simulation - Numba Accelerated
SBP-SAT finite difference method in velocity-stress form with parallel execution

Usage:
    python3 elasticwave2D_numba.py input.in [--saveplots] [--savefields] [--nthreads N]
"""

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from numba import njit, prange, set_num_threads, get_num_threads
import os

# Reload modules to pick up any changes
import importlib
modules_to_reload = ['RK4_2D', 'rate2d', 'first_derivative_sbp_operators', 
                     'boundarycondition', 'interface', 'interfacedata']
for mod in modules_to_reload:
    if mod in sys.modules:
        del sys.modules[mod]

import RK4_2D


def read_input_file(filename):
    """
    Read simulation parameters from input file.
    
    Parameters
    ----------
    filename : str
        Path to input configuration file
        
    Returns
    -------
    dict : Dictionary containing all simulation parameters
    """
    params = {}
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            
            # Parse key-value pairs
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.split('#')[0].strip()  # Remove inline comments
                
                # Try to convert to appropriate type
                try:
                    # Try integer first
                    params[key] = int(value)
                except ValueError:
                    try:
                        # Try float
                        params[key] = float(value)
                    except ValueError:
                        # Keep as string (remove quotes if present)
                        params[key] = value.strip('"').strip("'")
    
    return params


@njit(parallel=True)
def initialize_grids_numba(X_l, Y_l, X_r, Y_r, nx, ny, Lx, dx, dy):
    """Initialize spatial grids in parallel."""
    for i in prange(nx):
        for j in range(ny):
            X_l[i, j] = -Lx + i * dx
            Y_l[i, j] = j * dy
            X_r[i, j] = i * dx
            Y_r[i, j] = j * dy


@njit
def initialize_friction_params_numba(friction_parameters, alpha, Tau_0, L0, f0, a, b, V0, 
                                     sigma_n, alp_s, alp_d, D_c, ny):
    """Pack friction parameters efficiently."""
    for j in range(ny):
        friction_parameters[0, j] = alpha[j, 0]
        friction_parameters[1, j] = alpha[j, 0]
        friction_parameters[2, j] = Tau_0[j, 0]
        friction_parameters[3, j] = L0[j, 0]
        friction_parameters[4, j] = f0[j, 0]
        friction_parameters[5, j] = a[j, 0]
        friction_parameters[6, j] = b[j, 0]
        friction_parameters[7, j] = V0[j, 0]
        friction_parameters[8, j] = sigma_n[j, 0]
        friction_parameters[9, j] = alp_s[j, 0]
        friction_parameters[10, j] = alp_d[j, 0]
        friction_parameters[11, j] = D_c[j, 0]


def initialize_friction_parameters(params, ny, dy, Y0):
    """Initialize friction law parameters."""
    
    fric_law = params['fric_law']
    
    Y_fault = np.zeros((ny, 1))
    for j in range(ny):
        if np.abs(j*dy - Y0) <= 1.5:
            Y_fault[j, 0] = 1.0
    
    slip = np.zeros((ny, 1))
    psi = np.zeros((ny, 1))
    
    if fric_law == 'SW':
        # Slip-weakening friction
        slip = np.zeros((ny, 1))
        Tau_0 = np.ones((ny, 1)) * (70 + 11.6 * Y_fault)
        alp_s = np.ones((ny, 1)) * 0.677
        alp_d = np.ones((ny, 1)) * 0.525
        D_c = np.ones((ny, 1)) * 0.4
        sigma_n = -np.ones((ny, 1)) * 120.0
        
        psi = np.ones((ny, 1)) * 0.0
        L0 = np.ones((ny, 1)) * 1.0
        f0 = np.ones((ny, 1)) * 1.0
        a = np.ones((ny, 1)) * 1.0
        b = np.ones((ny, 1)) * 1.0
        V0 = np.ones((ny, 1)) * 1.0
        alpha = np.ones((ny, 1)) * 1e1000000
        
    elif fric_law == 'RS':
        # Rate-and-state friction
        alpha = np.ones((ny, 1)) * 1e1000000
        slip = np.ones((ny, 1)) * 0.0
        L0 = np.ones((ny, 1)) * 0.02
        f0 = np.ones((ny, 1)) * 0.6
        a = np.ones((ny, 1)) * 0.008
        b = np.ones((ny, 1)) * 0.012
        V0 = np.ones((ny, 1)) * 1.0e-6
        sigma_n = -np.ones((ny, 1)) * 120.0
        Tau_0 = np.ones((ny, 1)) * 75
        Vin = np.ones((ny, 1)) * 2.0e-12
        theta = L0 / V0 * np.exp(((a * np.log(2.0 * np.sinh(75 / (a * 120))) - f0 - a * np.log(Vin / V0)) / b))
        psi[:, 0] = f0[:, 0] + b[:, 0] * np.log(V0[:, 0] / L0[:, 0] * theta[:, 0])
        
        alp_s = np.ones((ny, 1)) * 1.0
        alp_d = np.ones((ny, 1)) * 1.0
        D_c = np.ones((ny, 1)) * 1.0
    
    else:
        raise ValueError(f"Unknown friction law: {fric_law}. Use 'RS' or 'SW'.")
    
    # Pack friction parameters using Numba
    friction_parameters = np.zeros((12, ny))
    initialize_friction_params_numba(friction_parameters, alpha, Tau_0, L0, f0, a, b, 
                                    V0, sigma_n, alp_s, alp_d, D_c, ny)
    
    return friction_parameters, slip, psi, sigma_n, Tau_0


@njit(parallel=True)
def store_domain_output(DomainOutput_l, DomainOutput_r, F_l, F_r, it):
    """Store domain outputs in parallel."""
    nx, ny, nf = F_l.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nf):
                DomainOutput_l[i, j, it, k] = F_l[i, j, k]
                DomainOutput_r[i, j, it, k] = F_r[i, j, k]


@njit(parallel=True)
def compute_slip_rate_spacetime(VT, FaultOutput, nt, ny):
    """Compute slip rate space-time data in parallel."""
    for it in prange(nt):
        for j in range(ny):
            VT[it, j] = np.sqrt(FaultOutput[j, it, 0]**2 + FaultOutput[j, it, 1]**2)


def run_simulation(params, nthreads=None):
    """
    Run the 2D elastic wave equation simulation with Numba acceleration.
    
    Parameters
    ----------
    params : dict
        Dictionary containing simulation parameters
    nthreads : int, optional
        Number of threads for parallel execution
        
    Returns
    -------
    tuple : Simulation results (FaultOutput, DomainOutput_l, DomainOutput_r, etc.)
    """
    
    # Set number of threads
    if nthreads is not None:
        set_num_threads(nthreads)
    
    actual_threads = get_num_threads()
    
    # Get system info
    import multiprocessing
    total_cores = multiprocessing.cpu_count()
    
    print("\n" + "="*60)
    print("STEP 1: SYSTEM & PARAMETERS")
    print("="*60)
    print(f"  System CPU cores: {total_cores}")
    print(f"  Numba parallel threads: {actual_threads}")
    if nthreads is None:
        print(f"  Threading: Auto (using all {actual_threads} available threads)")
    else:
        print(f"  Threading: User-specified ({nthreads} threads requested)")
    print()
    
    # Extract parameters
    Lx = params['Lx']
    Ly = params['Ly']
    nx = params['nx']
    ny = params['ny']
    tend = params['tend']
    CFL = params['CFL']
    order = params['order']
    
    cs = params['cs']
    cp = params['cp']
    rho = params['rho']
    
    x0 = params['x0']
    y0 = params['y0']
    t0 = params['t0']
    T = params['T']
    M0 = params['M0']
    source_type = params['source_type']
    
    fric_law = params['fric_law']
    mode = params.get('mode', 'II')
    model_type = params.get('model_type', 'homogeneous')
    
    Y0 = params.get('Y0', 10.0)
    isnap = params.get('isnap', 5)
    
    # Derived parameters
    dx = Lx / nx
    dy = Ly / ny
    
    if mode == 'II':
        nf = 5
    elif mode == 'III':
        nf = 3
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    M = [0, 0, 1., 1., 0]
    source_parameter = [x0, y0, t0, T, M0, source_type, M]
    
    # Extract Lame parameters
    mu = rho * cs**2
    Lambda = rho * cp**2 - 2.0 * mu
    
    dt = CFL / np.sqrt(cp**2 + cs**2) * dx
    nt = int(np.round(tend / dt))
    
    print(f"  Domain: Lx={Lx} km, Ly={Ly} km")
    print(f"  Grid: nx={nx}, ny={ny} ({nx*ny:,} points)")
    print(f"  Time: tend={tend} s, dt={dt:.6f} s, nt={nt}")
    print(f"  Order: {order}")
    print(f"  Friction law: {fric_law}")
    print(f"  Mode: {mode}")
    
    print("\n" + "="*60)
    print("STEP 2: INITIALIZING MATERIAL PROPERTIES")
    print("="*60)
    
    # Initialize velocity model
    Mat_l = np.zeros((nx, ny, 3))
    Mat_r = np.zeros((nx, ny, 3))
    
    if model_type == "homogeneous":
        print(f"  Model type: {model_type}")
        print(f"  Setting uniform properties: rho={rho}, Lambda={Lambda:.2f}, mu={mu:.2f}")
        Mat_l[:, :, 0] = rho
        Mat_l[:, :, 1] = Lambda
        Mat_l[:, :, 2] = mu
        
        Mat_r[:, :, 0] = rho
        Mat_r[:, :, 1] = Lambda
        Mat_r[:, :, 2] = mu
    
    elif model_type == "random":
        print(f"  Model type: {model_type}")
        print(f"  Generating random perturbations...")
        pert_l = 0.4
        r_rho_l = 2.0 * (np.random.rand(nx, ny) - 0.5) * pert_l
        r_mu_l = 2.0 * (np.random.rand(nx, ny) - 0.5) * pert_l
        r_lambda_l = 2.0 * (np.random.rand(nx, ny) - 0.5) * pert_l
        Mat_l[:, :, 0] = rho * (1.0 + r_rho_l)
        Mat_l[:, :, 1] = Lambda * (1.0 + r_lambda_l)
        Mat_l[:, :, 2] = mu * (1.0 + r_mu_l)
        
        pert_r = 0.4
        r_rho_r = 2.0 * (np.random.rand(nx, ny) - 0.5) * pert_r
        r_mu_r = 2.0 * (np.random.rand(nx, ny) - 0.5) * pert_r
        r_lambda_r = 2.0 * (np.random.rand(nx, ny) - 0.5) * pert_r
        Mat_r[:, :, 0] = rho * (1.0 + r_rho_r)
        Mat_r[:, :, 1] = Lambda * (1.0 + r_lambda_r)
        Mat_r[:, :, 2] = mu * (1.0 + r_mu_r)
    
    print("\n" + "="*60)
    print("STEP 3: INITIALIZING FIELDS AND GRIDS (Numba Parallel)")
    print("="*60)
    print(f"  Allocating arrays for {nf} fields...")
    
    # Initialize fields
    F_l = np.zeros((nx, ny, nf))
    Fnew_l = np.zeros((nx, ny, nf))
    X_l = np.zeros((nx, ny))
    Y_l = np.zeros((nx, ny))
    
    F_r = np.zeros((nx, ny, nf))
    Fnew_r = np.zeros((nx, ny, nf))
    X_r = np.zeros((nx, ny))
    Y_r = np.zeros((nx, ny))
    
    # Initialize grids using Numba parallel
    print(f"  Initializing grid coordinates in parallel...")
    initialize_grids_numba(X_l, Y_l, X_r, Y_r, nx, ny, Lx, dx, dy)
    print(f"  ✓ Grid coordinates initialized")
    
    print("\n" + "="*60)
    print("STEP 4: SETTING UP FRICTION PARAMETERS")
    print("="*60)
    
    # Initialize friction parameters
    friction_parameters, slip, psi, sigma_n, Tau_0 = initialize_friction_parameters(params, ny, dy, Y0)
    print(f"  Friction law '{fric_law}' configured for {ny} fault points")
    
    slip_new = np.zeros((ny, 1))
    psi_new = np.zeros((ny, 1))
    
    print("\n" + "="*60)
    print("STEP 5: ALLOCATING OUTPUT ARRAYS")
    print("="*60)
    
    # Initialize output arrays
    FaultOutput = np.zeros((ny, nt, 6))
    FaultOutput0 = np.zeros((ny, 6))
    
    fault_size = FaultOutput.nbytes / 1024**2
    print(f"  FaultOutput: ({ny} × {nt} × 6) = {fault_size:.1f} MB")
    
    FaultOutput[:, 0, 2] = sigma_n[:, 0]
    FaultOutput[:, 0, 3] = Tau_0[:, 0]
    FaultOutput[:, 0, 4] = slip[:, 0]
    FaultOutput[:, 0, 5] = psi[:, 0]
    
    print("\n" + "="*60)
    print("STEP 6: CONFIGURING RECEIVERS")
    print("="*60)
    
    DomainOutput_l = np.zeros((nx, ny, nt, nf))
    DomainOutput_r = np.zeros((nx, ny, nt, nf))
    
    domain_size = (DomainOutput_l.nbytes + DomainOutput_r.nbytes) / 1024**2
    print(f"  DomainOutput: 2 × ({nx} × {ny} × {nt} × {nf}) = {domain_size:.1f} MB")
    print(f"  Total memory: {(fault_size + domain_size):.1f} MB")
    
    # Receiver locations
    rx_l = np.array([-3.0])
    ry_l = np.array([0.0])
    irx_l = np.array([1])
    iry_l = np.array([0])
    
    for i in range(len(rx_l)):
        irx_l[i] = int((np.ceil(rx_l[i] / dx)) + (nx - 1))
        iry_l[i] = int(np.ceil(ry_l[i] / dy))
    
    rx_r = np.array([3.0])
    ry_r = np.array([1.0])
    irx_r = np.array([1])
    iry_r = np.array([0])
    
    for i in range(len(rx_r)):
        irx_r[i] = int(np.ceil(rx_r[i] / dx))
        iry_r[i] = int(np.ceil(ry_r[i] / dy))
    
    seisvx_l = np.zeros((len(irx_l), nt))
    seisvy_l = np.zeros((len(irx_l), nt))
    seisvx_r = np.zeros((len(irx_r), nt))
    seisvy_r = np.zeros((len(irx_r), nt))
    
    print(f"  Left receivers: {len(rx_l)} at positions {rx_l} km")
    print(f"  Right receivers: {len(rx_r)} at positions {rx_r} km")
    
    # Boundary reflection coefficients
    r_l = np.array([0., 0., 1., 0.])
    r_r = np.array([0., 0., 1., 0.])
    
    ir_l = np.arange(len(irx_l))
    ir_r = np.arange(len(irx_r))
    
    # Time-stepping loop
    print("\n" + "="*60)
    print("STEP 7: TIME-STEPPING SIMULATION (RK4 + Numba acceleration)")
    print("="*60)
    print(f"  Running {nt} time steps (dt={dt:.6f} s)...")
    print(f"  Using {actual_threads} parallel threads\n")
    
    import time
    t_start = time.time()
    
    # Progress reporting setup
    update_interval = max(1, nt // 100)  # Update every 1%
    print_interval = max(1, nt // 20)     # Print line every 5%
    
    print("  Progress:")
    print("  " + "-" * 56)
    
    for it in range(nt):
        t = it * dt
        
        # 4th order Runge-Kutta (serial in time, parallel in space via rate2d)
        RK4_2D.elastic_RK4_2D(
            Fnew_l, F_l, Mat_l, X_l, Y_l, t, nf, nx, ny, dx, dy, dt, order, r_l, source_parameter,
            Fnew_r, F_r, Mat_r, X_r, Y_r, r_r, friction_parameters, slip, psi, slip_new, psi_new,
            fric_law, FaultOutput0, Y0
        )
        
        # Update fields
        F_l = Fnew_l.copy()
        F_r = Fnew_r.copy()
        slip = slip_new.copy()
        psi = psi_new.copy()
        
        FaultOutput[:, it, :] = FaultOutput0
        
        # Store domain outputs using Numba parallel
        store_domain_output(DomainOutput_l, DomainOutput_r, F_l, F_r, it)
        
        # Save seismograms
        if mode == 'II':
            seisvx_l[ir_l, it] = F_l[irx_l[ir_l], iry_l[ir_l], 0]
            seisvy_l[ir_l, it] = F_l[irx_l[ir_l], iry_l[ir_l], 1]
            seisvx_r[ir_r, it] = F_r[irx_r[ir_r], iry_r[ir_r], 0]
            seisvy_r[ir_r, it] = F_r[irx_r[ir_r], iry_r[ir_r], 1]
        elif mode == 'III':
            seisvx_l[ir_l, it] = F_l[irx_l[ir_l], iry_l[ir_l], 0]
            seisvx_r[ir_r, it] = F_r[irx_r[ir_r], iry_r[ir_r], 0]
        
        # Progress reporting
        if it % update_interval == 0 or it == nt - 1:
            percent = 100 * (it + 1) / nt
            elapsed = time.time() - t_start
            rate = (it + 1) / elapsed if elapsed > 0 else 0
            eta = (nt - it - 1) / rate if rate > 0 else 0
            
            # Progress bar
            bar_width = 40
            filled = int(bar_width * (it + 1) / nt)
            bar = '█' * filled + '░' * (bar_width - filled)
            
            # Print with carriage return for live update
            if it % print_interval == 0 or it == nt - 1:
                print(f'  [{percent:5.1f}%] |{bar}| {it+1:4d}/{nt} | '
                      f't={t:.2f}s | {rate:.1f} it/s | ETA: {eta:.0f}s', flush=True)
    
    t_total = time.time() - t_start
    print("  " + "-" * 56)
    print(f"  ✓ Time-stepping complete: {t_total:.2f}s ({nt/t_total:.2f} it/s)")
    print("\n" + "="*60)
    print("TIME-STEPPING COMPLETE")
    print("="*60)
    
    # Return all results
    results = {
        'FaultOutput': FaultOutput,
        'DomainOutput_l': DomainOutput_l,
        'DomainOutput_r': DomainOutput_r,
        'seisvx_l': seisvx_l,
        'seisvy_l': seisvy_l,
        'seisvx_r': seisvx_r,
        'seisvy_r': seisvy_r,
        'X_l': X_l,
        'Y_l': Y_l,
        'X_r': X_r,
        'Y_r': Y_r,
        'rx_l': rx_l,
        'ry_l': ry_l,
        'rx_r': rx_r,
        'ry_r': ry_r,
        'dt': dt,
        'nt': nt,
        'nx': nx,
        'ny': ny,
        'Lx': Lx,
        'Ly': Ly,
        'dx': dx,
        'dy': dy,
        'x0': x0,
        'y0': y0,
        'mode': mode,
        'compute_time': t_total
    }
    
    return results


def save_plots(results, output_prefix='sim'):
    """
    Generate and save all plots.
    
    Parameters
    ----------
    results : dict
        Dictionary containing simulation results
    output_prefix : str
        Prefix for output filenames
    """
    
    print("\n" + "="*60)
    print("STEP 8: GENERATING PLOTS")
    print("="*60)
    
    FaultOutput = results['FaultOutput']
    DomainOutput_l = results['DomainOutput_l']
    DomainOutput_r = results['DomainOutput_r']
    seisvx_l = results['seisvx_l']
    seisvy_l = results['seisvy_l']
    seisvx_r = results['seisvx_r']
    seisvy_r = results['seisvy_r']
    Y_l = results['Y_l']
    rx_l = results['rx_l']
    ry_l = results['ry_l']
    rx_r = results['rx_r']
    ry_r = results['ry_r']
    dt = results['dt']
    nt = results['nt']
    Lx = results['Lx']
    Ly = results['Ly']
    x0 = results['x0']
    y0 = results['y0']
    mode = results['mode']
    
    # 1. Wave animation
    print("\n[1/5] Creating wave animation...")
    plt.ioff()
    fig_anim = plt.figure(figsize=(8, 6))
    ax_anim = fig_anim.add_subplot(111)
    
    if mode == 'II':
        p_l = DomainOutput_l[:, :, 0, 1]
        p_r = DomainOutput_r[:, :, 0, 1]
    elif mode == 'III':
        p_l = DomainOutput_l[:, :, 0, 0]
        p_r = DomainOutput_r[:, :, 0, 0]
    
    v = 2.0
    image_r = ax_anim.imshow(
        np.squeeze(np.append([p_l.transpose()], [p_r.transpose()], axis=2)),
        aspect='auto', extent=[-Lx, Lx, Ly, 0],
        cmap='seismic', vmin=-v, vmax=+v,
        interpolation='none'
    )
    
    for x, y in zip(rx_l, ry_l):
        ax_anim.text(x, y, '+', color='black', fontsize=12)
    for x, y in zip(rx_r, ry_r):
        ax_anim.text(x, y, '+', color='black', fontsize=12)
    ax_anim.text(x0, y0, 'o', color='red', fontsize=12)
    
    plt.colorbar(image_r, ax=ax_anim)
    ax_anim.set_xlabel('x [km]')
    ax_anim.set_ylabel('y [km]')
    
    def update_frame(frame_it):
        if mode == 'II':
            p_l = DomainOutput_l[:, :, frame_it, 1]
            p_r = DomainOutput_r[:, :, frame_it, 1]
        elif mode == 'III':
            p_l = DomainOutput_l[:, :, frame_it, 0]
            p_r = DomainOutput_r[:, :, frame_it, 0]
        
        p_b = np.squeeze(np.append([p_l.transpose()], [p_r.transpose()], axis=2))
        image_r.set_data(p_b)
        ax_anim.set_title(f"Time: {frame_it*dt:.2f} s")
        return [image_r]
    
    frames = range(0, nt - 1, 10)
    anim = FuncAnimation(fig_anim, update_frame, frames=frames, blit=True, interval=50)
    
    print(f"  Rendering animation ({len(frames)} frames)...")
    writer = PillowWriter(fps=20)
    anim.save(f'{output_prefix}_wave_animation.gif', writer=writer)
    print(f"  ✓ Saved: {output_prefix}_wave_animation.gif")
    del anim
    plt.close(fig_anim)
    
    # 2. Seismograms
    print("\n[2/5] Creating seismograms...")
    fig4 = plt.figure(figsize=(10, 4))
    time = np.arange(nt) * dt
    tend = nt * dt
    
    if mode == 'II':
        ay1 = fig4.add_subplot(2, 2, 1)
        ymax = seisvx_l.ravel().max()
        for ir_l in range(len(seisvx_l)):
            ay1.plot(time, seisvx_l[ir_l, :] + ymax * ir_l)
        ay1.set_ylabel('vx (m/s)')
        ay1.set_xlim([0, tend])
        ay1.set_ylim([-3, 3])
        ay1.set_title('Left side')
        
        ay2 = fig4.add_subplot(2, 2, 2)
        ymax = seisvy_l.ravel().max()
        for ir_l in range(len(seisvy_l)):
            ay2.plot(time, seisvy_l[ir_l, :] + ymax * ir_l)
        ay2.set_ylabel('vy (m/s)')
        ay2.set_xlim([0, tend])
        ay2.set_ylim([-3, 3])
        
        ay3 = fig4.add_subplot(2, 2, 3)
        ymax = seisvx_r.ravel().max()
        for ir_r in range(len(seisvx_r)):
            ay3.plot(time, seisvx_r[ir_r, :] + ymax * ir_r)
        ay3.set_xlabel('Time (s)')
        ay3.set_ylabel('vx (m/s)')
        ay3.set_xlim([0, tend])
        ay3.set_ylim([-3, 3])
        ay3.set_title('Right side')
        
        ay4 = fig4.add_subplot(2, 2, 4)
        ymax = seisvy_r.ravel().max()
        for ir_r in range(len(seisvy_r)):
            ay4.plot(time, seisvy_r[ir_r, :] + ymax * ir_r)
        ay4.set_xlabel('Time (s)')
        ay4.set_ylabel('vy (m/s)')
        ay4.set_xlim([0, tend])
        ay4.set_ylim([-3, 3])
    
    elif mode == 'III':
        plt.subplot(2, 1, 1)
        ymax = seisvx_l.ravel().max()
        for ir_l in range(len(seisvx_l)):
            plt.plot(time, seisvx_l[ir_l, :] + ymax * ir_l)
        plt.xlabel('Time (s)')
        plt.ylabel('vx (m/s)')
        plt.title('Left side')
        
        plt.subplot(2, 1, 2)
        ymax = seisvx_r.ravel().max()
        for ir_r in range(len(seisvx_r)):
            plt.plot(time, seisvx_r[ir_r, :] + ymax * ir_r)
        plt.xlabel('Time (s)')
        plt.ylabel('vx (m/s)')
        plt.title('Right side')
    
    fig4.tight_layout()
    fig4.savefig(f'{output_prefix}_seismograms.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_prefix}_seismograms.png")
    plt.close(fig4)
    
    # 3. Fault evolution animation
    print("\n[3/5] Creating fault evolution animation...")
    fig1 = plt.figure(figsize=(10, 10))
    ax3 = fig1.add_subplot(3, 1, 1)
    ax3.set_ylabel('Slip [m]')
    ax3.set_ylim([0, 20])
    ax3.set_xlim([0, Ly])
    
    ax4 = fig1.add_subplot(3, 1, 2)
    ax4.set_ylabel('Slip rate [m/s]')
    ax4.set_ylim([0, 7])
    ax4.set_xlim([0, Ly])
    
    ax5 = fig1.add_subplot(3, 1, 3)
    ax5.set_xlabel('Fault [km]')
    ax5.set_ylabel('Stress [MPa]')
    ax5.set_ylim([50, 90])
    ax5.set_xlim([0, Ly])
    
    plt.tight_layout()
    
    y_fault = Y_l[-1, :]
    
    line3, = ax3.plot([], [], 'g', lw=2, label='slip')
    line4, = ax4.plot([], [], 'g', lw=2, label='slip rate')
    line5, = ax5.plot([], [], 'g', lw=2, label='traction')
    ax3.legend()
    ax4.legend()
    ax5.legend()
    
    def update_fault(frame_it):
        slip_ = FaultOutput[:, frame_it, 4]
        sliprate_ = np.sqrt(FaultOutput[:, frame_it, 0]**2 + FaultOutput[:, frame_it, 1]**2)
        traction_ = FaultOutput[:, frame_it, 3]
        
        line3.set_data(y_fault, slip_)
        line4.set_data(y_fault, sliprate_)
        line5.set_data(y_fault, traction_)
        fig1.suptitle(f'On-Fault Evolution - Time: {frame_it*dt:.2f} s')
        return line3, line4, line5
    
    frames_fault = range(0, nt, 5)
    anim_fault = FuncAnimation(fig1, update_fault, frames=frames_fault, blit=True, interval=50)
    
    print(f"  Rendering animation ({len(frames_fault)} frames)...")
    writer = PillowWriter(fps=20)
    anim_fault.save(f'{output_prefix}_fault_evolution.gif', writer=writer)
    print(f"  ✓ Saved: {output_prefix}_fault_evolution.gif")
    
    # Save final state
    update_fault(nt - 1)
    fig1.savefig(f'{output_prefix}_fault_evolution_final.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_prefix}_fault_evolution_final.png")
    del anim_fault
    plt.close(fig1)
    
    # 4. Slip rate space-time plot
    print("\n[4/5] Creating slip rate space-time plot (Numba accelerated)...")
    ny = FaultOutput.shape[0]
    VT = np.zeros((nt, ny))
    
    # Compute using Numba parallel
    compute_slip_rate_spacetime(VT, FaultOutput, nt, ny)
    
    fig_slip_time = plt.figure(figsize=(10, 6))
    v = 2.5
    image = plt.imshow(
        VT, aspect='auto', extent=[0, Ly, nt * dt, 0],
        cmap='viridis', vmin=0, vmax=+v, interpolation='none'
    )
    
    plt.colorbar(label='Slip rate [m/s]')
    plt.xlabel('Fault [km]')
    plt.ylabel('Time [s]')
    plt.title('Slip Rate Space-Time Evolution')
    
    fig_slip_time.savefig(f'{output_prefix}_slip_rate_spacetime.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_prefix}_slip_rate_spacetime.png")
    plt.close(fig_slip_time)
    
    print("\n" + "="*60)
    print("ALL PLOTS SAVED")
    print("="*60)


def save_fields(results, output_prefix='sim'):
    """
    Save all field data to .npz file.
    
    Parameters
    ----------
    results : dict
        Dictionary containing simulation results
    output_prefix : str
        Prefix for output filename
    """
    
    print("\n" + "="*60)
    print("STEP 9: SAVING FIELD DATA")
    print("="*60)
    print(f"  Compressing and writing data...")
    
    filename = f'{output_prefix}_fields.npz'
    
    np.savez_compressed(
        filename,
        FaultOutput=results['FaultOutput'],
        DomainOutput_l=results['DomainOutput_l'],
        DomainOutput_r=results['DomainOutput_r'],
        seisvx_l=results['seisvx_l'],
        seisvy_l=results['seisvy_l'],
        seisvx_r=results['seisvx_r'],
        seisvy_r=results['seisvy_r'],
        X_l=results['X_l'],
        Y_l=results['Y_l'],
        X_r=results['X_r'],
        Y_r=results['Y_r'],
        dt=results['dt'],
        nt=results['nt'],
        nx=results['nx'],
        ny=results['ny'],
        Lx=results['Lx'],
        Ly=results['Ly'],
        compute_time=results['compute_time']
    )
    
    file_size = os.path.getsize(filename) / 1024**2
    print(f"  ✓ Saved: {filename} ({file_size:.1f} MB)")
    print("\n" + "="*60)
    print("FIELD DATA SAVED")
    print("="*60)


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description='2D Elastic Wave Equation - Numba Parallel Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 elasticwave2D_numba.py default.in
  python3 elasticwave2D_numba.py default.in --saveplots --nthreads 8
  python3 elasticwave2D_numba.py highres.in --saveplots --savefields --nthreads 16
        """
    )
    
    parser.add_argument('input_file', help='Input configuration file')
    parser.add_argument('--saveplots', action='store_true', help='Save all plots (GIF and PNG)')
    parser.add_argument('--savefields', action='store_true', help='Save all fields as .npz')
    parser.add_argument('--output-prefix', default='sim', help='Output filename prefix (default: sim)')
    parser.add_argument('--nthreads', type=int, default=None, 
                       help='Number of Numba parallel threads (default: auto)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("2D ELASTIC WAVE EQUATION - NUMBA PARALLEL")
    print("="*60)
    print(f"Input file: {args.input_file}")
    print(f"Output prefix: {args.output_prefix}")
    print(f"Save plots: {args.saveplots}")
    print(f"Save fields: {args.savefields}")
    if args.nthreads:
        print(f"Threads requested: {args.nthreads}")
    else:
        print(f"Threads: auto (will use all available)")
    
    # Read input file
    print("\nReading configuration...")
    params = read_input_file(args.input_file)
    print(f"✓ Loaded {len(params)} parameters\n")
    
    # Display key parameters
    print("Key Input Parameters:")
    print("  " + "-" * 56)
    for key in ['Lx', 'Ly', 'nx', 'ny', 'tend', 'CFL', 'order', 'cs', 'cp', 'rho', 'fric_law', 'mode']:
        if key in params:
            print(f"  {key:15s} = {params[key]}")
    print("  " + "-" * 56)
    
    # Run simulation
    results = run_simulation(params, args.nthreads)
    
    # Save outputs
    if args.saveplots:
        save_plots(results, args.output_prefix)
    
    if args.savefields:
        save_fields(results, args.output_prefix)
    
    if not args.saveplots and not args.savefields:
        print("\nNote: Use --saveplots to save visualizations, --savefields to save data")
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE!")
    print(f"Total compute time: {results['compute_time']:.2f} seconds")
    print("="*60)


if __name__ == '__main__':
    main()
