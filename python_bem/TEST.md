# Python BEM Solver - Quick Test

This script runs a quick test to verify the installation.

```python
import numpy as np
from bem_solver import BEMSolver

print("Testing BEM Solver Installation...")
print("=" * 50)

# Create a simple circular fiber
radius = 1e-6
wavelength = 600e-9
c0 = 299792458.0

solver = BEMSolver.from_circle(
    numel=50,  # Small for quick test
    radius=radius,
    mode='TM',
    freq=c0/wavelength,
    epsr=2.25  # n=1.5
)

print(f"✓ Solver created successfully")
print(f"  - Radius: {radius*1e6:.2f} μm")
print(f"  - Elements: {solver.numel}")
print(f"  - Wavelength: {wavelength*1e9:.1f} nm")

# Solve
sol = solver.solve()
print(f"✓ System solved successfully")
print(f"  - Solution size: {len(sol)}")
print(f"  - Max current: {np.max(np.abs(sol)):.4e}")

# Compute energy balance
energy = solver.compute_energy_balance()
print(f"✓ Energy balance computed: {energy:.6f}")

# Compute scattering
sigma = solver.compute_bsdf(180, 1000*radius)
print(f"✓ BSDF computed successfully")
print(f"  - Total scattering: {np.sum(sigma):.4e}")

print("=" * 50)
print("All tests passed! Installation is working correctly.")
```

Save this as `test_installation.py` and run:

```bash
python test_installation.py
```
