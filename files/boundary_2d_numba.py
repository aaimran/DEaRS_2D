"""
Numba-optimized 2D boundary conditions for elastic and acoustic waves
SAT penalty method implementation
"""

import numpy as np
from numba import njit, prange
import math

# ============================================================================
# BOUNDARY CONDITIONS - X DIRECTION
# ============================================================================

@njit(fastmath=True, cache=True)
def bcm2dx_elastic_numba(BF, F, Mat, nx, ny, r0x):
    """
    Left boundary (x=0) for elastic waves
    
    Args:
        BF: Output SAT terms (ny, 5)
        F: Fields (nx, ny, 5) - vx, vy, sxx, syy, sxy
        Mat: Material (nx, ny, 3) - rho, Lambda, mu
        nx, ny: Grid dimensions
        r0x: Reflection coefficient
    """
    for j in prange(ny):
        # Material parameters
        rho = Mat[0, j, 0]
        Lambda = Mat[0, j, 1]
        mu = Mat[0, j, 2]
        
        # Wave speeds
        cp = math.sqrt((2.0*mu + Lambda) / rho)
        cs = math.sqrt(mu / rho)
        
        # Impedances
        zp = rho * cp
        zs = rho * cs
        
        # Boundary fields
        vx = F[0, j, 0]
        vy = F[0, j, 1]
        sxx = F[0, j, 2]
        syy = F[0, j, 3]
        sxy = F[0, j, 4]
        
        # Characteristics
        px = 0.5 * (zp*vx - sxx)
        py = 0.5 * (zs*vy - sxy)
        qx = 0.5 * (zp*vx + sxx)
        qy = 0.5 * (zs*vy + sxy)
        
        # SAT terms
        BF[j, 0] = 1.0/rho * (px - r0x*qx)
        BF[j, 1] = 1.0/rho * (py - r0x*qy)
        BF[j, 2] = -(2.0*mu + Lambda)/zp * (px - r0x*qx)
        BF[j, 3] = -Lambda/zp * (px - r0x*qx)
        BF[j, 4] = -mu/zs * (py - r0x*qy)


@njit(fastmath=True, cache=True)
def bcp2dx_elastic_numba(BF, F, Mat, nx, ny, rnx):
    """Right boundary (x=Lx) for elastic waves"""
    for j in prange(ny):
        # Material parameters
        rho = Mat[nx-1, j, 0]
        Lambda = Mat[nx-1, j, 1]
        mu = Mat[nx-1, j, 2]
        
        # Wave speeds
        cp = math.sqrt((2.0*mu + Lambda) / rho)
        cs = math.sqrt(mu / rho)
        
        # Impedances
        zp = rho * cp
        zs = rho * cs
        
        # Boundary fields
        vx = F[nx-1, j, 0]
        vy = F[nx-1, j, 1]
        sxx = F[nx-1, j, 2]
        syy = F[nx-1, j, 3]
        sxy = F[nx-1, j, 4]
        
        # Characteristics
        px = 0.5 * (zp*vx + sxx)
        py = 0.5 * (zs*vy + sxy)
        qx = 0.5 * (zp*vx - sxx)
        qy = 0.5 * (zs*vy - sxy)
        
        # SAT terms
        BF[j, 0] = 1.0/rho * (px - rnx*qx)
        BF[j, 1] = 1.0/rho * (py - rnx*qy)
        BF[j, 2] = (2.0*mu + Lambda)/zp * (px - rnx*qx)
        BF[j, 3] = Lambda/zp * (px - rnx*qx)
        BF[j, 4] = mu/zs * (py - rnx*qy)


# ============================================================================
# BOUNDARY CONDITIONS - Y DIRECTION
# ============================================================================

@njit(fastmath=True, cache=True)
def bcm2dy_elastic_numba(BF, F, Mat, nx, ny, r0y):
    """Bottom boundary (y=0) for elastic waves"""
    for i in prange(nx):
        # Material parameters
        rho = Mat[i, 0, 0]
        Lambda = Mat[i, 0, 1]
        mu = Mat[i, 0, 2]
        
        # Wave speeds
        cp = math.sqrt((2.0*mu + Lambda) / rho)
        cs = math.sqrt(mu / rho)
        
        # Impedances
        zp = rho * cp
        zs = rho * cs
        
        # Boundary fields
        vx = F[i, 0, 0]
        vy = F[i, 0, 1]
        sxx = F[i, 0, 2]
        syy = F[i, 0, 3]
        sxy = F[i, 0, 4]
        
        # Characteristics
        px = 0.5 * (zs*vx - sxy)
        py = 0.5 * (zp*vy - syy)
        qx = 0.5 * (zs*vx + sxy)
        qy = 0.5 * (zp*vy + syy)
        
        # SAT terms
        BF[i, 0] = 1.0/rho * (px - r0y*qx)
        BF[i, 1] = 1.0/rho * (py - r0y*qy)
        BF[i, 2] = -Lambda/zp * (py - r0y*qy)
        BF[i, 3] = -(2.0*mu + Lambda)/zp * (py - r0y*qy)
        BF[i, 4] = -mu/zs * (px - r0y*qx)


@njit(fastmath=True, cache=True)
def bcp2dy_elastic_numba(BF, F, Mat, nx, ny, rny):
    """Top boundary (y=Ly) for elastic waves"""
    for i in prange(nx):
        # Material parameters
        rho = Mat[i, ny-1, 0]
        Lambda = Mat[i, ny-1, 1]
        mu = Mat[i, ny-1, 2]
        
        # Wave speeds
        cp = math.sqrt((2.0*mu + Lambda) / rho)
        cs = math.sqrt(mu / rho)
        
        # Impedances
        zp = rho * cp
        zs = rho * cs
        
        # Boundary fields
        vx = F[i, ny-1, 0]
        vy = F[i, ny-1, 1]
        sxx = F[i, ny-1, 2]
        syy = F[i, ny-1, 3]
        sxy = F[i, ny-1, 4]
        
        # Characteristics
        px = 0.5 * (zs*vx + sxy)
        py = 0.5 * (zp*vy + syy)
        qx = 0.5 * (zs*vx - sxy)
        qy = 0.5 * (zp*vy - syy)
        
        # SAT terms
        BF[i, 0] = 1.0/rho * (px - rny*qx)
        BF[i, 1] = 1.0/rho * (py - rny*qy)
        BF[i, 2] = Lambda/zp * (py - rny*qy)
        BF[i, 3] = (2.0*mu + Lambda)/zp * (py - rny*qy)
        BF[i, 4] = mu/zs * (px - rny*qx)


# ============================================================================
# INTERFACE COUPLING (FAULT)
# ============================================================================

@njit(fastmath=True, cache=True)
def interface_fault_left_numba(BF, F, Fhat, Mat, nx, ny):
    """
    Interface penalty for left side of fault
    
    Args:
        BF: Output SAT terms (ny, 5)
        F: Current fields (nx, ny, 5)
        Fhat: Interface hat-variables (ny, 4) - Vx, Vy, Sxx, Sxy
        Mat: Material properties (nx, ny, 3)
        nx, ny: Grid dimensions
    """
    for j in prange(ny):
        # Extract fields at interface (right edge of left block)
        vx = F[nx-1, j, 0]
        vy = F[nx-1, j, 1]
        sxx = F[nx-1, j, 2]
        sxy = F[nx-1, j, 4]
        
        # Hat variables
        Vx = Fhat[j, 0]
        Vy = Fhat[j, 1]
        Sxx = Fhat[j, 2]
        Sxy = Fhat[j, 3]
        
        # Material parameters
        rho = Mat[nx-1, j, 0]
        Lambda = Mat[nx-1, j, 1]
        mu = Mat[nx-1, j, 2]
        
        # Wave speeds and impedances
        cp = math.sqrt((2.0*mu + Lambda) / rho)
        cs = math.sqrt(mu / rho)
        zp = rho * cp
        zs = rho * cs
        
        # Characteristics
        px = 0.5 * (zp*vx + sxx)
        py = 0.5 * (zs*vy + sxy)
        
        Px = 0.5 * (zp*Vx + Sxx)
        Py = 0.5 * (zs*Vy + Sxy)
        
        # SAT terms
        BF[j, 0] = 1.0/rho * (px - Px)
        BF[j, 1] = 1.0/rho * (py - Py)
        BF[j, 2] = (2.0*mu + Lambda)/zp * (px - Px)
        BF[j, 3] = Lambda/zp * (px - Px)
        BF[j, 4] = mu/zs * (py - Py)


@njit(fastmath=True, cache=True)
def interface_fault_right_numba(BF, F, Fhat, Mat, nx, ny):
    """Interface penalty for right side of fault"""
    for j in prange(ny):
        # Extract fields at interface (left edge of right block)
        vx = F[0, j, 0]
        vy = F[0, j, 1]
        sxx = F[0, j, 2]
        sxy = F[0, j, 4]
        
        # Hat variables
        Vx = Fhat[j, 0]
        Vy = Fhat[j, 1]
        Sxx = Fhat[j, 2]
        Sxy = Fhat[j, 3]
        
        # Material parameters
        rho = Mat[0, j, 0]
        Lambda = Mat[0, j, 1]
        mu = Mat[0, j, 2]
        
        # Wave speeds and impedances
        cp = math.sqrt((2.0*mu + Lambda) / rho)
        cs = math.sqrt(mu / rho)
        zp = rho * cp
        zs = rho * cs
        
        # Characteristics
        px = 0.5 * (zp*vx - sxx)
        py = 0.5 * (zs*vy - sxy)
        
        Px = 0.5 * (zp*Vx - Sxx)
        Py = 0.5 * (zs*Vy - Sxy)
        
        # SAT terms
        BF[j, 0] = 1.0/rho * (px - Px)
        BF[j, 1] = 1.0/rho * (py - Py)
        BF[j, 2] = -(2.0*mu + Lambda)/zp * (px - Px)
        BF[j, 3] = -Lambda/zp * (px - Px)
        BF[j, 4] = -mu/zs * (py - Py)


# ============================================================================
# PENALTY WEIGHTS
# ============================================================================

@njit(fastmath=True, cache=True)
def penalty_weights_2d_numba(order, dx):
    """Compute SAT penalty weights for 2D"""
    if order == 2:
        return 0.5 * dx
    elif order == 4:
        return (17.0 / 48.0) * dx
    elif order == 6:
        return 13649.0 / 43200.0 * dx
    else:
        return 0.5 * dx
