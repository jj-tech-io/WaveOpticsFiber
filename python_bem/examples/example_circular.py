"""
Example: Scattering from a circular fiber.

This example demonstrates how to use the BEM solver to simulate
electromagnetic scattering from a circular cross-section fiber.
"""
import sys
import os
# Add parent directory to path to import bem_solver
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from bem_solver import BEMSolver

def main():
    print("=" * 60)
    print("BEM Scattering Simulation - Circular Fiber")
    print("=" * 60)
    
    # Fiber parameters
    radius = 1e-6  # 1 micrometer
    n_elements = 100  # Number of boundary elements
    
    # Material properties
    n_fiber = 1.55  # Refractive index of fiber (e.g., glass)
    epsr = n_fiber**2  # Relative permittivity
    
    # Wave parameters
    wavelength = 600e-9  # 600 nm (orange light)
    c0 = 299792458.0
    freq = c0 / wavelength
    theta = np.pi / 2  # Perpendicular to cylinder axis
    phi_i = 0.0  # Incident azimuthal angle
    
    print(f"\nFiber radius: {radius*1e6:.2f} μm")
    print(f"Refractive index: {n_fiber}")
    print(f"Wavelength: {wavelength*1e9:.1f} nm")
    print(f"Number of elements: {n_elements}")
    print(f"Incident angle: θ={np.degrees(theta):.1f}°, φ={np.degrees(phi_i):.1f}°")
    
    # Create solver for TM mode
    print("\n" + "-" * 60)
    print("Solving TM mode...")
    solver_tm = BEMSolver.from_circle(
        numel=n_elements,
        radius=radius,
        quadrature=2,
        phi_i=phi_i,
        mode='TM',
        freq=freq,
        mur=1.0,
        epsr=epsr,
        theta=theta
    )
    
    # Solve the system
    print("Assembling system matrix...")
    sol_tm = solver_tm.solve()
    print(f"Solution computed. Max current: {np.max(np.abs(sol_tm)):.4e}")
    
    # Compute energy balance
    energy_tm = solver_tm.compute_energy_balance()
    print(f"Energy balance (TM): {energy_tm:.6f}")
    
    # Compute scattering distribution
    n_angles = 360
    distance = 1000 * radius  # Far field distance
    print(f"\nComputing BSDF at {n_angles} angles...")
    sigma_tm = solver_tm.compute_bsdf(n_angles, distance)
    
    # Solve for TE mode
    print("\n" + "-" * 60)
    print("Solving TE mode...")
    solver_te = BEMSolver.from_circle(
        numel=n_elements,
        radius=radius,
        quadrature=2,
        phi_i=phi_i,
        mode='TE',
        freq=freq,
        mur=1.0,
        epsr=epsr,
        theta=theta
    )
    
    sol_te = solver_te.solve()
    print(f"Solution computed. Max current: {np.max(np.abs(sol_te)):.4e}")
    
    energy_te = solver_te.compute_energy_balance()
    print(f"Energy balance (TE): {energy_te:.6f}")
    
    sigma_te = solver_te.compute_bsdf(n_angles, distance)
    
    # Average TM and TE
    sigma_avg = (sigma_tm + sigma_te) / 2
    
    # Plot results
    print("\n" + "-" * 60)
    print("Plotting results...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Polar plot of scattering pattern
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    ax1 = plt.subplot(121, projection='polar')
    ax1.plot(angles, sigma_tm, 'b-', label='TM mode', linewidth=1.5)
    ax1.plot(angles, sigma_te, 'r-', label='TE mode', linewidth=1.5)
    ax1.plot(angles, sigma_avg, 'k-', label='Average', linewidth=2)
    ax1.set_theta_zero_location('E')
    ax1.set_title('Scattering Pattern (BSDF)', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax1.grid(True)
    
    # Plot 2: Cartesian plot
    ax2 = plt.subplot(122)
    ax2.plot(np.degrees(angles), sigma_tm, 'b-', label='TM mode', linewidth=1.5)
    ax2.plot(np.degrees(angles), sigma_te, 'r-', label='TE mode', linewidth=1.5)
    ax2.plot(np.degrees(angles), sigma_avg, 'k-', label='Average', linewidth=2)
    ax2.set_xlabel('Scattering Angle (degrees)', fontsize=12)
    ax2.set_ylabel('Scattered Intensity', fontsize=12)
    ax2.set_title('Scattering Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 360)
    
    plt.tight_layout()
    plt.savefig('circular_fiber_scattering.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'circular_fiber_scattering.png'")
    plt.show()
    
    # Summary
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Total scattering (TM): {np.sum(sigma_tm):.4e}")
    print(f"Total scattering (TE): {np.sum(sigma_te):.4e}")
    print(f"Total scattering (Avg): {np.sum(sigma_avg):.4e}")
    print(f"Energy balance (TM): {energy_tm:.6f}")
    print(f"Energy balance (TE): {energy_te:.6f}")
    print(f"Forward scattering (180°): {sigma_avg[n_angles//2]:.4e}")
    print(f"Backward scattering (0°): {sigma_avg[0]:.4e}")
    print("=" * 60)

if __name__ == "__main__":
    main()
