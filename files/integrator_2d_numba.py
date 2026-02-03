"""
Numba-optimized 2D elastic wave rate computation and RK4 time integration
Includes fault interface with friction laws
"""

import numpy as np
from numba import njit, prange
import math

# Import from 1D implementation (reuse friction laws)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'numba'))
from friction_numba import regula_falsi_numba, FRICTION_RS, FRICTION_SW, FRICTION_LN, FRICTION_LOCKED

from operators_2d_numba import dx2d_numba, dy2d_numba
from boundary_2d_numba import (bcm2dx_elastic_numba, bcp2dx_elastic_numba,
                               bcm2dy_elastic_numba, bcp2dy_elastic_numba,
                               interface_fault_left_numba, interface_fault_right_numba,
                               penalty_weights_2d_numba)

# ============================================================================
# INTERFACE CONDITIONS WITH FRICTION
# ============================================================================

@njit(fastmath=True, cache=True)
def interface_condition_2d_numba(vx_m, Tx_m, vx_p, Tx_p, rho_m, mu_eff_m, rho_p, mu_eff_p):
    """
    Locked interface condition (normal direction)
    
    Args:
        vx_m, Tx_m: Velocity and traction (minus side)
        vx_p, Tx_p: Velocity and traction (plus side)
        rho_m, mu_eff_m: Material properties (minus)
        rho_p, mu_eff_p: Material properties (plus)
    
    Returns:
        V_m, V_p, T_m, T_p: Interface velocities and tractions
    """
    # Wave speeds
    cs_m = math.sqrt(mu_eff_m / rho_m)
    Zs_m = rho_m * cs_m
    
    cs_p = math.sqrt(mu_eff_p / rho_p)
    Zs_p = rho_p * cs_p
    
    # Characteristics
    q_m = Zs_m * vx_m - Tx_m
    p_p = Zs_p * vx_p + Tx_p
    
    # Harmonic mean impedance
    eta_s = Zs_m * Zs_p / (Zs_m + Zs_p)
    
    # Stress transfer
    Phi = eta_s * (p_p/Zs_p - q_m/Zs_m)
    
    # Locked condition
    vv = 0.0
    T_m = Phi
    T_p = Phi
    
    V_m = (p_p - T_p)/Zs_p - vv
    V_p = (q_m + T_m)/Zs_m + vv
    
    return V_m, V_p, T_m, T_p


@njit(fastmath=True, cache=True)
def friction_2d_numba(vy_m, Ty_m, vy_p, Ty_p, slip, psi,
                     rho_m, mu_m, rho_p, mu_p,
                     friction_type, alpha, Tau_0, L0, f0, a, b, V0,
                     sigma_n, alp_s, alp_d, D_c):
    """
    Friction law for tangential (y) direction
    Reuses 1D friction law logic
    """
    # Wave speeds
    cs_m = math.sqrt(mu_m / rho_m)
    Zs_m = rho_m * cs_m
    
    cs_p = math.sqrt(mu_p / rho_p)
    Zs_p = rho_p * cs_p
    
    # Characteristics
    q_m = Zs_m * vy_m - Ty_m
    p_p = Zs_p * vy_p + Ty_p
    
    # Harmonic mean
    eta_s = Zs_m * Zs_p / (Zs_m + Zs_p)
    
    # Stress transfer
    Phi = eta_s * (p_p/Zs_p - q_m/Zs_m)
    
    # ========================================================================
    # LOCKED
    # ========================================================================
    if friction_type == FRICTION_LOCKED:
        vv = 0.0
        T_m = Phi
        T_p = Phi
        V_m = (p_p - T_p)/Zs_p - vv
        V_p = (q_m + T_m)/Zs_m + vv
        return V_m, V_p, T_m, T_p
    
    # ========================================================================
    # RATE-AND-STATE
    # ========================================================================
    elif friction_type == FRICTION_RS:
        # Initialize
        V_abs = abs(vy_p - vy_m)
        if V_abs > Phi:
            V = 0.5 * Phi / eta_s
        else:
            V = V_abs
        
        # Solve
        vv = regula_falsi_numba(V, Phi + Tau_0, eta_s, sigma_n, psi, V0, a)
        
        # Traction
        Tau_h = (Phi + Tau_0) - eta_s * vv
        
        T_m = Tau_h - Tau_0
        T_p = Tau_h - Tau_0
        
        V_m = (p_p - T_p)/Zs_p - vv
        V_p = (q_m + T_m)/Zs_m + vv
        
        return V_m, V_p, T_m, T_p
    
    # ========================================================================
    # SLIP-WEAKENING
    # ========================================================================
    elif friction_type == FRICTION_SW:
        Tau_lock = Phi + Tau_0
        
        # Strength
        if slip < D_c:
            fric_coeff = alp_s - (alp_s - alp_d) * slip / D_c
        else:
            fric_coeff = alp_d
        Tau_str = fric_coeff * sigma_n
        
        if Tau_lock >= Tau_str:
            # Slipping
            vv = ((Phi + Tau_0) - Tau_str) / eta_s
            Tau_h = (Phi + Tau_0) - eta_s * vv
            
            T_m = Tau_h - Tau_0
            T_p = Tau_h - Tau_0
            
            V_m = (p_p - T_p)/Zs_p - vv
            V_p = (q_m + T_m)/Zs_m + vv
        else:
            # Locked
            vv = 0.0
            T_m = Phi
            T_p = Phi
            V_m = (p_p - T_p)/Zs_p - vv
            V_p = (q_m + T_m)/Zs_m + vv
        
        return V_m, V_p, T_m, T_p
    
    # ========================================================================
    # LINEAR VISCOUS
    # ========================================================================
    elif friction_type == FRICTION_LN:
        coeff = 1.0 / (eta_s/alpha + 1.0)
        T_m = coeff * Phi
        T_p = coeff * Phi
        vv = Phi / (eta_s + alpha)
        
        V_m = (p_p - T_p)/Zs_p - vv
        V_p = (q_m + T_m)/Zs_m + vv
        
        return V_m, V_p, T_m, T_p
    
    # Default: locked
    else:
        vv = 0.0
        T_m = Phi
        T_p = Phi
        V_m = (p_p - T_p)/Zs_p - vv
        V_p = (q_m + T_m)/Zs_m + vv
        return V_m, V_p, T_m, T_p


@njit(fastmath=True, cache=True)
def compute_interface_2d_numba(F_l, F_r, Fhat_l, Fhat_r, Mat_l, Mat_r,
                               nx, ny, friction_params, slip, psi,
                               dslip, dpsi, friction_type, Y, t, Y0):
    """
    Compute interface hat-variables with friction
    
    Args:
        F_l, F_r: Fields (nx, ny, 5) for left and right blocks
        Fhat_l, Fhat_r: Output hat-variables (ny, 4)
        Mat_l, Mat_r: Materials (nx, ny, 3)
        nx, ny: Grid dimensions
        friction_params: Friction parameters (12, ny)
        slip, psi: Interface state (ny, 1)
        dslip, dpsi: Output rates (ny, 1)
        friction_type: Friction law type
        Y: Y-coordinates (ny,)
        t: Time
        Y0: Nucleation center
    """
    for j in prange(ny):
        # Nucleation perturbation (for rupture initiation)
        r = math.sqrt((Y[j] - Y0)**2)
        
        F = 0.0
        if r < 3.0:
            F = math.exp(r**2 / (r**2 - 9.0))
        
        G = 0.0
        if t > 0.0 and t < 1.0:
            G = math.exp((t - 1.0)**2 / (t * (t - 2.0)))
        elif t >= 1.0:
            G = 1.0
        
        # Extract parameters
        tau_base = friction_params[2, j]
        if friction_type == FRICTION_RS:
            tau = tau_base + 25.0 * F * G
        else:
            tau = tau_base
        
        # Material properties
        rho_l = Mat_l[nx-1, j, 0]
        lam_l = Mat_l[nx-1, j, 1]
        mu_l = Mat_l[nx-1, j, 2]
        
        rho_r = Mat_r[0, j, 0]
        lam_r = Mat_r[0, j, 1]
        mu_r = Mat_r[0, j, 2]
        
        twomulam_l = 2.0*mu_l + lam_l
        twomulam_r = 2.0*mu_r + lam_r
        
        # Extract interface fields
        vx_l = F_l[nx-1, j, 0]
        vy_l = F_l[nx-1, j, 1]
        Tx_l = F_l[nx-1, j, 2]
        Ty_l = F_l[nx-1, j, 4]
        
        vx_r = F_r[0, j, 0]
        vy_r = F_r[0, j, 1]
        Tx_r = F_r[0, j, 2]
        Ty_r = F_r[0, j, 4]
        
        # Normal direction: locked interface
        vx_m, vx_p, Tx_m, Tx_p = interface_condition_2d_numba(
            vx_l, Tx_l, vx_r, Tx_r,
            rho_l, twomulam_l, rho_r, twomulam_r
        )
        
        Fhat_l[j, 0] = vx_m
        Fhat_r[j, 0] = vx_p
        Fhat_l[j, 2] = Tx_m
        Fhat_r[j, 2] = Tx_p
        
        # Compute normal stress
        sigma_n = max(0.0, -(Tx_m + friction_params[8, j]))
        
        # Tangential direction: friction
        L0 = friction_params[3, j]
        f0 = friction_params[4, j]
        a = friction_params[5, j]
        b = friction_params[6, j]
        V0 = friction_params[7, j]
        alp_s = friction_params[9, j]
        alp_d = friction_params[10, j]
        D_c = friction_params[11, j]
        alpha = friction_params[1, j]
        
        vy_m, vy_p, Ty_m, Ty_p = friction_2d_numba(
            vy_l, Ty_l, vy_r, Ty_r,
            slip[j, 0], psi[j, 0],
            rho_l, mu_l, rho_r, mu_r,
            friction_type, alpha, tau, L0, f0, a, b, V0,
            sigma_n, alp_s, alp_d, D_c
        )
        
        Fhat_l[j, 1] = vy_m
        Fhat_r[j, 1] = vy_p
        Fhat_l[j, 3] = Ty_m
        Fhat_r[j, 3] = Ty_p
        
        # Slip rate
        vv = abs(vy_p - vy_m)
        dslip[j, 0] = vv
        
        # State evolution
        if friction_type == FRICTION_RS:
            dpsi[j, 0] = b*V0/L0 * math.exp(-(psi[j, 0] - f0)/b) - vv*b/L0
        else:
            dpsi[j, 0] = 0.0


# ============================================================================
# ELASTIC RATE COMPUTATION
# ============================================================================

@njit(fastmath=True, cache=True)
def elastic_rate2d_numba(D_l, F_l, Mat_l, nx, ny, dx, dy, order, r_l,
                        D_r, F_r, Mat_r, r_r,
                        friction_params, slip, psi, dslip, dpsi,
                        friction_type, Y, t, Y0):
    """
    Compute elastic rates for both blocks with fault coupling
    
    Args:
        D_l: Output rates (nx, ny, 5) for left block
        F_l: Current fields (nx, ny, 5) for left block
        Mat_l: Material (nx, ny, 3) for left block
        nx, ny, dx, dy: Grid parameters
        order: Spatial order
        r_l: Boundary reflection coefficients [r0x, rnx, r0y, rny] for left
        (similar for right block)
        friction_params: Friction parameters (12, ny)
        slip, psi: Interface state (ny, 1)
        dslip, dpsi: Output interface rates (ny, 1)
        friction_type: Friction law type
        Y: Y-coordinates
        t: Time
        Y0: Nucleation center
    """
    nf = 5
    
    # Allocate temporary arrays
    dxF_l = np.zeros(nf)
    dyF_l = np.zeros(nf)
    dxF_r = np.zeros(nf)
    dyF_r = np.zeros(nf)
    
    # Compute interior rates
    for i in prange(nx):
        for j in range(ny):
            # Left block
            rho_l = Mat_l[i, j, 0]
            lam_l = Mat_l[i, j, 1]
            mu_l = Mat_l[i, j, 2]
            
            # Spatial derivatives
            dx2d_numba(dxF_l, F_l, nx, i, j, dx, order)
            dy2d_numba(dyF_l, F_l, ny, i, j, dy, order)
            
            # Momentum equation
            D_l[i, j, 0] = 1.0/rho_l * (dxF_l[2] + dyF_l[4])
            D_l[i, j, 1] = 1.0/rho_l * (dxF_l[4] + dyF_l[3])
            
            # Hooke's law
            D_l[i, j, 2] = (2.0*mu_l + lam_l)*dxF_l[0] + lam_l*dyF_l[1]
            D_l[i, j, 3] = (2.0*mu_l + lam_l)*dyF_l[1] + lam_l*dxF_l[0]
            D_l[i, j, 4] = mu_l * (dyF_l[0] + dxF_l[1])
            
            # Right block
            rho_r = Mat_r[i, j, 0]
            lam_r = Mat_r[i, j, 1]
            mu_r = Mat_r[i, j, 2]
            
            # Spatial derivatives
            dx2d_numba(dxF_r, F_r, nx, i, j, dx, order)
            dy2d_numba(dyF_r, F_r, ny, i, j, dy, order)
            
            # Momentum equation
            D_r[i, j, 0] = 1.0/rho_r * (dxF_r[2] + dyF_r[4])
            D_r[i, j, 1] = 1.0/rho_r * (dxF_r[4] + dyF_r[3])
            
            # Hooke's law
            D_r[i, j, 2] = (2.0*mu_r + lam_r)*dxF_r[0] + lam_r*dyF_r[1]
            D_r[i, j, 3] = (2.0*mu_r + lam_r)*dyF_r[1] + lam_r*dxF_r[0]
            D_r[i, j, 4] = mu_r * (dyF_r[0] + dxF_r[1])
    
    # Compute interface hat-variables
    Fhat_l = np.zeros((ny, 4))
    Fhat_r = np.zeros((ny, 4))
    
    compute_interface_2d_numba(F_l, F_r, Fhat_l, Fhat_r, Mat_l, Mat_r,
                              nx, ny, friction_params, slip, psi,
                              dslip, dpsi, friction_type, Y, t, Y0)
    
    # Impose boundaries with SAT
    hx = penalty_weights_2d_numba(order, dx)
    hy = penalty_weights_2d_numba(order, dy)
    
    # Boundary SAT terms
    BF0x_l = np.zeros((ny, nf))
    BFnx_l = np.zeros((ny, nf))
    BF0y_l = np.zeros((nx, nf))
    BFny_l = np.zeros((nx, nf))
    
    BF0x_r = np.zeros((ny, nf))
    BFnx_r = np.zeros((ny, nf))
    BF0y_r = np.zeros((nx, nf))
    BFny_r = np.zeros((nx, nf))
    
    # Left block boundaries
    bcm2dx_elastic_numba(BF0x_l, F_l, Mat_l, nx, ny, r_l[0])
    interface_fault_left_numba(BFnx_l, F_l, Fhat_l, Mat_l, nx, ny)
    bcm2dy_elastic_numba(BF0y_l, F_l, Mat_l, nx, ny, r_l[2])
    bcp2dy_elastic_numba(BFny_l, F_l, Mat_l, nx, ny, r_l[3])
    
    # Right block boundaries
    interface_fault_right_numba(BF0x_r, F_r, Fhat_r, Mat_r, nx, ny)
    bcp2dx_elastic_numba(BFnx_r, F_r, Mat_r, nx, ny, r_r[1])
    bcm2dy_elastic_numba(BF0y_r, F_r, Mat_r, nx, ny, r_r[2])
    bcp2dy_elastic_numba(BFny_r, F_r, Mat_r, nx, ny, r_r[3])
    
    # Apply SAT penalties
    for j in range(ny):
        for k in range(nf):
            D_l[0, j, k] -= BF0x_l[j, k] / hx
            D_l[nx-1, j, k] -= BFnx_l[j, k] / hx
            
            D_r[0, j, k] -= BF0x_r[j, k] / hx
            D_r[nx-1, j, k] -= BFnx_r[j, k] / hx
    
    for i in range(nx):
        for k in range(nf):
            D_l[i, 0, k] -= BF0y_l[i, k] / hy
            D_l[i, ny-1, k] -= BFny_l[i, k] / hy
            
            D_r[i, 0, k] -= BF0y_r[i, k] / hy
            D_r[i, ny-1, k] -= BFny_r[i, k] / hy


# ============================================================================
# RK4 TIME INTEGRATION
# ============================================================================

@njit(cache=True)
def elastic_RK4_2d_numba(F_l, F_r, Mat_l, Mat_r, nx, ny, dx, dy, dt, order,
                        r_l, r_r, friction_params, slip, psi,
                        friction_type, Y, t, Y0):
    """
    4th-order Runge-Kutta time integration for 2D elastic waves with fault
    
    Returns:
        Updates F_l, F_r, slip, psi in place
    """
    nf = 5
    
    # RK4 stages
    k1_l = np.zeros((nx, ny, nf))
    k2_l = np.zeros((nx, ny, nf))
    k3_l = np.zeros((nx, ny, nf))
    k4_l = np.zeros((nx, ny, nf))
    
    k1_r = np.zeros((nx, ny, nf))
    k2_r = np.zeros((nx, ny, nf))
    k3_r = np.zeros((nx, ny, nf))
    k4_r = np.zeros((nx, ny, nf))
    
    k1slip = np.zeros((ny, 1))
    k2slip = np.zeros((ny, 1))
    k3slip = np.zeros((ny, 1))
    k4slip = np.zeros((ny, 1))
    
    k1psi = np.zeros((ny, 1))
    k2psi = np.zeros((ny, 1))
    k3psi = np.zeros((ny, 1))
    k4psi = np.zeros((ny, 1))
    
    # Stage 1
    elastic_rate2d_numba(k1_l, F_l, Mat_l, nx, ny, dx, dy, order, r_l,
                        k1_r, F_r, Mat_r, r_r,
                        friction_params, slip, psi, k1slip, k1psi,
                        friction_type, Y, t, Y0)
    
    # Stage 2
    elastic_rate2d_numba(k2_l, F_l + 0.5*dt*k1_l, Mat_l, nx, ny, dx, dy, order, r_l,
                        k2_r, F_r + 0.5*dt*k1_r, Mat_r, r_r,
                        friction_params, slip + 0.5*dt*k1slip, psi + 0.5*dt*k1psi,
                        k2slip, k2psi, friction_type, Y, t + 0.5*dt, Y0)
    
    # Stage 3
    elastic_rate2d_numba(k3_l, F_l + 0.5*dt*k2_l, Mat_l, nx, ny, dx, dy, order, r_l,
                        k3_r, F_r + 0.5*dt*k2_r, Mat_r, r_r,
                        friction_params, slip + 0.5*dt*k2slip, psi + 0.5*dt*k2psi,
                        k3slip, k3psi, friction_type, Y, t + 0.5*dt, Y0)
    
    # Stage 4
    elastic_rate2d_numba(k4_l, F_l + dt*k3_l, Mat_l, nx, ny, dx, dy, order, r_l,
                        k4_r, F_r + dt*k3_r, Mat_r, r_r,
                        friction_params, slip + dt*k3slip, psi + dt*k3psi,
                        k4slip, k4psi, friction_type, Y, t + dt, Y0)
    
    # Update
    F_l[:, :, :] = F_l + (dt/6.0) * (k1_l + 2.0*k2_l + 2.0*k3_l + k4_l)
    F_r[:, :, :] = F_r + (dt/6.0) * (k1_r + 2.0*k2_r + 2.0*k3_r + k4_r)
    
    slip[:, :] = slip + (dt/6.0) * (k1slip + 2.0*k2slip + 2.0*k3slip + k4slip)
    psi[:, :] = psi + (dt/6.0) * (k1psi + 2.0*k2psi + 2.0*k3psi + k4psi)
