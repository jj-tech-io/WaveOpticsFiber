"""
Visualization utilities for BEM solver results.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_geometry(solver, ax=None, show_elements=True, show_currents=False, solution=None):
    """
    Plot the fiber geometry.
    
    Parameters:
        solver: BEMSolver instance
        ax: Matplotlib axis (creates new if None)
        show_elements: If True, mark element boundaries
        show_currents: If True, color by current magnitude (requires solution)
        solution: Solution vector for current visualization
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    nodes = solver.nodes
    x = nodes[:, 0] * 1e6  # Convert to micrometers
    y = nodes[:, 1] * 1e6
    
    if show_currents and solution is not None:
        # Color by current magnitude
        n = solver.numel
        current_mag = np.abs(solution[:n])
        scatter = ax.scatter(x, y, c=current_mag, cmap='hot', s=50, zorder=5)
        plt.colorbar(scatter, ax=ax, label='|J_t|')
    
    # Plot boundary
    x_closed = np.append(x, x[0])
    y_closed = np.append(y, y[0])
    ax.plot(x_closed, y_closed, 'b-', linewidth=2)
    ax.fill(x_closed, y_closed, alpha=0.2, color='blue')
    
    if show_elements:
        ax.plot(x, y, 'ro', markersize=4)
    
    ax.set_aspect('equal')
    ax.set_xlabel('x (μm)', fontsize=12)
    ax.set_ylabel('y (μm)', fontsize=12)
    ax.set_title('Fiber Cross-Section', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_scattering_polar(angles, sigma, ax=None, title='Scattering Pattern'):
    """
    Plot scattering pattern in polar coordinates.
    
    Parameters:
        angles: Array of angles in radians
        sigma: Array of scattering intensities
        ax: Matplotlib polar axis
        title: Plot title
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 8))
    
    ax.plot(angles, sigma, 'k-', linewidth=2)
    ax.fill(angles, sigma, alpha=0.3)
    ax.set_theta_zero_location('E')
    ax.set_title(title, pad=20, fontsize=14)
    ax.grid(True)
    
    return ax

def plot_scattering_cartesian(angles, sigma, ax=None):
    """
    Plot scattering pattern in Cartesian coordinates.
    
    Parameters:
        angles: Array of angles in radians
        sigma: Array of scattering intensities
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(np.degrees(angles), sigma, 'k-', linewidth=2)
    ax.fill_between(np.degrees(angles), sigma, alpha=0.3)
    ax.set_xlabel('Scattering Angle (degrees)', fontsize=12)
    ax.set_ylabel('Scattered Intensity', fontsize=12)
    ax.set_title('Scattering Distribution', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)
    
    # Mark forward and backward scattering
    ax.axvline(0, color='r', linestyle='--', alpha=0.5, label='Backward')
    ax.axvline(180, color='g', linestyle='--', alpha=0.5, label='Forward')
    ax.legend()
    
    return ax

def plot_comparison(angles, data_dict, title='Comparison', polar=True):
    """
    Plot multiple scattering patterns for comparison.
    
    Parameters:
        angles: Array of angles
        data_dict: Dictionary {label: sigma_array}
        title: Plot title
        polar: If True, use polar plot
    """
    if polar:
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(10, 10))
        for label, sigma in data_dict.items():
            ax.plot(angles, sigma, linewidth=2, label=label)
        ax.set_theta_zero_location('E')
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        for label, sigma in data_dict.items():
            ax.plot(np.degrees(angles), sigma, linewidth=2, label=label)
        ax.set_xlabel('Scattering Angle (degrees)', fontsize=12)
        ax.set_ylabel('Scattered Intensity', fontsize=12)
        ax.set_xlim(0, 360)
    
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax

def plot_full_results(solver, solution, sigma, angles):
    """
    Create comprehensive visualization of results.
    
    Parameters:
        solver: BEMSolver instance
        solution: Solution vector
        sigma: BSDF array
        angles: Array of angles
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Geometry with currents
    ax1 = plt.subplot(2, 3, 1)
    plot_geometry(solver, ax=ax1, show_currents=True, solution=solution)
    
    # Polar scattering
    ax2 = plt.subplot(2, 3, 2, projection='polar')
    plot_scattering_polar(angles, sigma, ax=ax2)
    
    # Cartesian scattering
    ax3 = plt.subplot(2, 3, 3)
    plot_scattering_cartesian(angles, sigma, ax=ax3)
    
    # Current components
    n = solver.numel
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(np.abs(solution[:n]), label='|J_t|')
    ax4.plot(np.abs(solution[n:2*n]), label='|J_z|')
    ax4.plot(np.abs(solution[2*n:3*n]), label='|M_t|')
    ax4.plot(np.abs(solution[3*n:4*n]), label='|M_z|')
    ax4.set_xlabel('Element index', fontsize=12)
    ax4.set_ylabel('Current magnitude', fontsize=12)
    ax4.set_title('Surface Currents', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Log-scale scattering
    ax5 = plt.subplot(2, 3, 5)
    ax5.semilogy(np.degrees(angles), sigma + 1e-10, 'k-', linewidth=2)
    ax5.set_xlabel('Scattering Angle (degrees)', fontsize=12)
    ax5.set_ylabel('Scattered Intensity (log scale)', fontsize=12)
    ax5.set_title('Scattering (Log Scale)', fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, 360)
    
    # Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    energy = solver.compute_energy_balance()
    stats_text = f"""
    Simulation Statistics:
    ━━━━━━━━━━━━━━━━━━━━━━
    Elements: {solver.numel}
    Wavelength: {solver.wave.wavelength*1e9:.1f} nm
    Radius: {solver.radius*1e6:.2f} μm
    Mode: {solver.wave.mode}
    
    Results:
    ━━━━━━━━━━━━━━━━━━━━━━
    Max current: {np.max(np.abs(solution)):.4e}
    Energy balance: {energy:.6f}
    Total scattering: {np.sum(sigma):.4e}
    Peak angle: {np.degrees(angles[np.argmax(sigma)]):.1f}°
    Forward/Backward: {sigma[len(sigma)//2]/sigma[0]:.2f}
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    return fig

def save_results(filename, solver, solution, sigma, angles):
    """
    Save results to file.
    
    Parameters:
        filename: Base filename (without extension)
        solver: BEMSolver instance
        solution: Solution vector
        sigma: BSDF array
        angles: Array of angles
    """
    # Save numerical data
    np.savez(f"{filename}.npz",
             solution=solution,
             sigma=sigma,
             angles=angles,
             nodes=solver.nodes,
             wavelength=solver.wave.wavelength,
             radius=solver.radius)
    
    # Save plot
    fig = plot_full_results(solver, solution, sigma, angles)
    fig.savefig(f"{filename}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Results saved to {filename}.npz and {filename}.png")
