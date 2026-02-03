# 2D Elastic Wave Equation Simulation - Usage Guide

## Overview

This simulation code solves the 2D elastic wave equation with dynamic earthquake rupture using the SBP-SAT finite difference method. The code has been refactored from Jupyter notebooks into a standalone Python script for robustness and flexibility.

## Features

- **Configuration file input**: All parameters stored in `.in` files
- **Command-line interface**: Easy control of output options
- **High/low resolution modes**: Choose speed vs accuracy
- **Complete visualization**: All plots from both original notebooks
- **Data export**: Save full field data for post-processing

## Installation

Ensure you have the required dependencies:
```bash
pip3 install numpy matplotlib
```

The following local modules are required in the same directory:
- `RK4_2D.py`
- `rate2d.py`
- `interfacedata.py`
- `first_derivative_sbp_operators.py`
- `boundarycondition.py`
- `interface.py`

## Usage

### Basic syntax
```bash
python3 elasticwave2D.py <input_file> [options]
```

### Options
- `--saveplots`: Save all plots as GIF animations and PNG images
- `--savefields`: Save all field data (domain and fault) as compressed .npz file
- `--output-prefix PREFIX`: Set output filename prefix (default: 'sim')

### Examples

**1. Run with default parameters, no output files:**
```bash
python3 elasticwave2D.py default.in
```

**2. High-resolution simulation with plots:**
```bash
python3 elasticwave2D.py default.in --saveplots
```

**3. Low-resolution simulation with plots and field data:**
```bash
python3 elasticwave2D.py lowres.in --saveplots --savefields
```

**4. Custom output prefix:**
```bash
python3 elasticwave2D.py default.in --saveplots --output-prefix highres_test
```

## Input File Format

Input files use a simple `key = value` format. Lines starting with `#` are comments.

### Required parameters

```ini
# Domain
Lx = 10.0               # Domain length [km]
Ly = 20.0               # Domain width [km]

# Grid
nx = 100                # Grid points in x
ny = 200                # Grid points in y

# Time
tend = 10.0             # Simulation end time [s]
CFL = 0.5               # CFL number

# Material
cs = 3.464              # Shear wave speed [km/s]
cp = 6.0                # P-wave speed [km/s]
rho = 2.6702            # Density [g/cm^3]

# Source
x0 = -15.0              # Source location x [km]
y0 = 7.5                # Source location y [km]
t0 = 0.0                # Source time [s]
T = 0.1                 # Source duration [s]
M0 = 0.0                # Source moment [MPa]
source_type = Gaussian  # Gaussian or Brune

# Numerics
order = 6               # Spatial accuracy order (4 or 6)

# Friction
fric_law = RS           # RS (rate-state) or SW (slip-weakening)
mode = II               # Fracture mode: II or III

# Model
model_type = homogeneous  # homogeneous or random
Y0 = 10.0               # Nucleation patch center [km]
```

## Output Files

### Plots (`--saveplots`)

1. **`{prefix}_wave_animation.gif`**: Animated wave propagation through the domain
   - Shows particle velocity field
   - Samples every 10th time step
   - 20 fps animation

2. **`{prefix}_seismograms.png`**: Seismograms at receiver locations
   - Left and right domain receivers
   - vx and vy components (Mode II)
   - High resolution (300 dpi)

3. **`{prefix}_fault_evolution.gif`**: Animated on-fault behavior
   - Slip, slip rate, and traction evolution
   - Samples every 5th time step
   - 20 fps animation

4. **`{prefix}_fault_evolution_final.png`**: Final fault state
   - Slip, slip rate, and traction profiles
   - High resolution (300 dpi)

5. **`{prefix}_slip_rate_spacetime.png`**: Space-time plot of slip rate
   - Shows rupture propagation along fault
   - Time vs position visualization
   - High resolution (300 dpi)

### Field Data (`--savefields`)

**`{prefix}_fields.npz`**: Compressed numpy archive containing:
- `FaultOutput`: On-fault data (ny × nt × 6)
  - Columns: vx, vy, sigma_n, tau, slip, psi
- `DomainOutput_l`: Left domain fields (nx × ny × nt × nf)
- `DomainOutput_r`: Right domain fields (nx × ny × nt × nf)
- `seisvx_l`, `seisvy_l`: Left receiver seismograms
- `seisvx_r`, `seisvy_r`: Right receiver seismograms
- `X_l`, `Y_l`, `X_r`, `Y_r`: Grid coordinates
- `dt`, `nt`, `nx`, `ny`, `Lx`, `Ly`: Simulation parameters

### Loading saved data
```python
import numpy as np

# Load the data
data = np.load('sim_fields.npz')

# Access arrays
fault_data = data['FaultOutput']
domain_left = data['DomainOutput_l']
dt = data['dt']

# List all arrays
print(data.files)
```

## Configuration Comparison

### High Resolution (default.in)
- **Grid**: 100 × 200 (20,000 points)
- **Order**: 6th order spatial accuracy
- **Speed**: Slower, more accurate
- **Use**: Publication-quality results, detailed analysis

### Low Resolution (lowres.in)
- **Grid**: 26 × 51 (1,326 points)
- **Order**: 4th order spatial accuracy
- **Speed**: ~15× faster
- **Use**: Testing, parameter exploration, quick runs

## Friction Laws

### Rate-and-State (RS)
- More realistic earthquake physics
- Velocity weakening behavior
- State variable evolution
- Nucleation via stress perturbation

### Slip-Weakening (SW)
- Simpler model
- Linear strength reduction with slip
- Critical slip distance Dc
- Static/dynamic friction coefficients

## Tips

1. **Start with low resolution** for parameter testing:
   ```bash
   python3 elasticwave2D.py lowres.in --saveplots
   ```

2. **Save fields only when needed** (large files):
   - 100×200 grid, 500 time steps ≈ 500 MB compressed

3. **Monitor progress**: The code prints time-stepping updates every `isnap` iterations

4. **Check CFL condition**: Increase if simulation is unstable, decrease for better accuracy

5. **Memory usage**: High resolution with field saving requires ~2-3 GB RAM

## Troubleshooting

**Error: "Module not found"**
- Ensure all local modules (RK4_2D.py, etc.) are in the same directory

**Error: "friction law not implemented"**
- Check that `fric_law = RS` or `fric_law = SW` in input file

**Warning: Simulation unstable**
- Reduce CFL number in input file
- Increase grid resolution

**Plots not showing**
- This is expected! Plots are saved to files, not displayed interactively
- Use `--saveplots` flag

## Performance

Approximate runtime on modern laptop (M1/i7):

| Configuration | Grid | Time Steps | Runtime | Memory |
|--------------|------|------------|---------|---------|
| lowres.in | 26×51 | ~190 | 30 sec | 200 MB |
| default.in | 100×200 | ~500 | 8 min | 1.5 GB |

## Citation

If you use this code, please cite:

Duru, K., and E. M. Dunham (2016), Dynamic earthquake rupture simulations on nonplanar faults embedded in 3D geometrically complex, heterogeneous elastic solids, *J. Comput. Phys.*, 305, 185-207, doi:10.1016/j.jcp.2015.10.021
