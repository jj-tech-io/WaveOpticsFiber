"""
BEM Solver for 3D Wave Optics Scattering Problems

A Python implementation of the Boundary Element Method (BEM) for solving
electromagnetic scattering from infinite cylinders with arbitrary cross-sections.

Based on the SIGGRAPH Asia 2020 paper:
"A Wave Optics Based Fiber Scattering Model"
by Mengqi (Mandy) Xia, Bruce Walter, Eric Michielssen, David Bindel and Steve Marschner
"""

from .background import Background
from .wave import Wave
from .element import Element
from .basis import Basis
from .solver import BEMSolver

__version__ = "1.0.0"
__all__ = ['Background', 'Wave', 'Element', 'Basis', 'BEMSolver']

# Optional visualization utilities
try:
    from . import visualization
    __all__.append('visualization')
except ImportError:
    pass  # matplotlib not installed
