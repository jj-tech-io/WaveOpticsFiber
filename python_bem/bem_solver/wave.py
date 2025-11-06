"""
Wave parameters for incident electromagnetic field.
"""
import numpy as np

class Wave:
    """
    Incident wave properties and propagation parameters.
    
    Attributes:
        phi_i (float): Azimuthal incident angle (radians)
        mode (str): Polarization mode ('TM' or 'TE')
        freq (float): Frequency (Hz)
        wavelength (float): Wavelength in background medium (m)
        k0 (float): Wavenumber in background medium (rad/m)
        newk (float): Transverse wavenumber component (rad/m)
        kz (float): Longitudinal wavenumber component (rad/m)
        omega (float): Angular frequency (rad/s)
        wavelength1 (complex): Wavelength in scatterer (m)
        k1 (complex): Wavenumber in scatterer (rad/m)
        newk1 (complex): Transverse wavenumber in scatterer (rad/m)
    """
    
    def __init__(self, background, phi_i, mode, freq, theta):
        """
        Initialize wave parameters.
        
        Parameters:
            background (Background): Background and scatterer material properties
            phi_i (float): Azimuthal incident angle (radians)
            mode (str): Polarization mode ('TM' for TM mode, 'TE' for TE mode)
            freq (float): Frequency (Hz)
            theta (float): Longitudinal incident angle from z-axis (radians)
                          theta = pi/2 means perpendicular to cylinder axis
        """
        self.phi_i = phi_i
        self.mode = mode
        self.freq = freq
        self.omega = 2 * np.pi * freq
        
        # Background medium wave parameters
        self.wavelength = background.c0 / freq
        self.k0 = 2 * np.pi / self.wavelength
        self.newk = self.k0 * np.sin(theta)  # Transverse component
        self.kz = self.k0 * np.cos(theta)    # Longitudinal component
        
        # Scatterer wave parameters
        if np.abs(background.c1) > 0:
            self.wavelength1 = background.c1 / freq
            self.k1 = 2 * np.pi / self.wavelength1
            self.newk1 = np.sqrt(self.k1**2 - self.k0**2 * np.cos(theta)**2)
        else:
            self.wavelength1 = 0j
            self.k1 = 0j
            self.newk1 = 0j
    
    def __repr__(self):
        return (f"Wave(freq={self.freq:.2e} Hz, wavelength={self.wavelength*1e9:.1f} nm, "
                f"mode={self.mode}, phi_i={np.degrees(self.phi_i):.1f}°)")
