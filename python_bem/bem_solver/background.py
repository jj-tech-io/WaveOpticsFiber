"""
Background medium and scatterer material properties.
"""
import numpy as np

class Background:
    """
    Material properties for background medium and scatterer.
    
    Attributes:
        c0 (float): Speed of light in vacuum (m/s)
        mu0 (float): Permeability of free space (H/m)
        eps0 (float): Permittivity of free space (F/m)
        Z0 (float): Impedance of free space (Ω)
        mu1 (float): Permeability of scatterer (H/m)
        eps1 (complex): Permittivity of scatterer (F/m)
        c1 (complex): Speed of light in scatterer (m/s)
        Z1 (complex): Impedance of scatterer (Ω)
    """
    
    def __init__(self, mur=1.0, epsr=1.0+0j):
        """
        Initialize background and scatterer properties.
        
        Parameters:
            mur (float): Relative permeability of scatterer (dimensionless)
            epsr (complex): Relative permittivity of scatterer (dimensionless)
                           Can be complex for lossy materials
        """
        # Background (vacuum/air) properties
        self.c0 = 299792458.0  # Speed of light in vacuum (m/s)
        self.mu0 = 4e-7 * np.pi  # Permeability of free space (H/m)
        self.eps0 = 8.8541878e-12  # Permittivity of free space (F/m)
        self.Z0 = np.sqrt(self.mu0 / self.eps0)  # Impedance of free space (Ω)
        
        # Scatterer properties
        self.mu1 = self.mu0 * mur
        self.eps1 = self.eps0 * epsr
        self.c1 = 1.0 / np.sqrt(self.mu1 * self.eps1)
        self.Z1 = np.sqrt(self.mu1 / self.eps1)
    
    def __repr__(self):
        return (f"Background(c0={self.c0}, Z0={self.Z0:.2f}, "
                f"c1={self.c1}, Z1={self.Z1})")
