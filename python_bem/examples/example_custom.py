"""
Example: Scattering from an arbitrary cross-section fiber.

This example shows how to define a custom fiber cross-section
from coordinate arrays.
"""
import sys
import os
# Add parent directory to path to import bem_solver
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from bem_solver import BEMSolver

def create_custom_shape(n_points=100):
    """
    Create a custom fiber shape - a rounded square.
    
    Returns:
        np.ndarray: Array of node positions
    """
    # Create a rounded square using a superellipse formula
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    # Superellipse: |x/a|^n + |y/b|^n = 1
    a, b = 1.5e-6, 1.5e-6
    n = 3  # Higher n makes it more square
    
    x = a * np.sign(np.cos(t)) * np.abs(np.cos(t))**(2/n)
    y = b * np.sign(np.sin(t)) * np.abs(np.sin(t))**(2/n)
    
    nodes = np.zeros((n_points, 3))
    nodes[:, 0] = x
    nodes[:, 1] = y
    
    return nodes

def main():
    print("=" * 60)
    print("BEM Scattering Simulation - Custom Fiber Shape")
    print("=" * 60)
    
    # Create custom shape
    n_points = 120
    nodes = create_custom_shape(n_points)
    
    print(f"\nCustom shape with {n_points} boundary nodes")
    
    # Material properties
    n_fiber = 1.55
    epsr = n_fiber**2
    
    # Wave parameters
    wavelength = 600e-9  # 600 nm
    c0 = 299792458.0
    freq = c0 / wavelength
    theta = np.pi / 2
    phi_i = 0.0
    
    print(f"Refractive index: {n_fiber}")
    print(f"Wavelength: {wavelength*1e9:.1f} nm")
    
    # Create solver
    print("\n" + "-" * 60)
    print("Creating solver...")
    solver = BEMSolver(
        nodes=nodes,
        quadrature=2,
        phi_i=phi_i,
        mode='TM',
        freq=freq,
        mur=1.0,
        epsr=epsr,
        theta=theta
    )
    
    print(f"Effective radius: {solver.radius*1e6:.2f} μm")
    
    # Solve
    print("\nSolving BEM system...")
    sol = solver.solve()
    print(f"Solution computed. Max current: {np.max(np.abs(sol)):.4e}")
    
    # Energy balance
    energy = solver.compute_energy_balance()
    print(f"Energy balance: {energy:.6f}")
    
    # Compute BSDF
    n_angles = 360
    distance = 1000 * solver.radius
    print(f"\nComputing BSDF...")
    sigma = solver.compute_bsdf(n_angles, distance)
    
    # Plot results
    print("\nPlotting results...")
    fig = plt.figure(figsize=(14, 6))
    
    # Plot 1: Fiber geometry with surface currents
    ax1 = plt.subplot(131)
    
    # Plot boundary
    x = nodes[:, 0] * 1e6
    y = nodes[:, 1] * 1e6
    ax1.plot(x, y, 'b-', linewidth=2)
    ax1.fill(x, y, alpha=0.2, color='blue')
    
    # Plot current magnitude on boundary
    n = len(nodes)
    current_mag = np.abs(sol[:n])
    scatter = ax1.scatter(x, y, c=current_mag, cmap='hot', s=20, zorder=5)
    plt.colorbar(scatter, ax=ax1, label='|Current|')
    
    ax1.set_aspect('equal')
    ax1.set_xlabel('x (μm)', fontsize=12)
    ax1.set_ylabel('y (μm)', fontsize=12)
    ax1.set_title('Fiber Geometry & Surface Current', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Polar scattering pattern
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    ax2 = plt.subplot(132, projection='polar')
    ax2.plot(angles, sigma, 'k-', linewidth=2)
    ax2.set_theta_zero_location('E')
    ax2.set_title('Scattering Pattern', pad=20, fontsize=14)
    ax2.grid(True)
    
    # Plot 3: Cartesian scattering distribution
    ax3 = plt.subplot(133)
    ax3.plot(np.degrees(angles), sigma, 'k-', linewidth=2)
    ax3.set_xlabel('Scattering Angle (degrees)', fontsize=12)
    ax3.set_ylabel('Scattered Intensity', fontsize=12)
    ax3.set_title('Scattering Distribution', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 360)
    
    plt.tight_layout()
    plt.savefig('custom_fiber_scattering.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'custom_fiber_scattering.png'")
    plt.show()
    
    # Summary
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Total scattering: {np.sum(sigma):.4e}")
    print(f"Energy balance: {energy:.6f}")
    print(f"Max scattering angle: {np.degrees(angles[np.argmax(sigma)]):.1f}°")
    print(f"Max scattered intensity: {np.max(sigma):.4e}")
    print("=" * 60)

if __name__ == "__main__":
    main()
