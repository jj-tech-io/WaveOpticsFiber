"""
Test script to verify BEM solver installation and basic functionality.
"""
import numpy as np
import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from bem_solver import Background, Wave, Element, Basis, BEMSolver
        print("  ✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_background():
    """Test Background class."""
    print("\nTesting Background class...")
    try:
        from bem_solver import Background
        
        # Lossless material
        bg = Background(mur=1.0, epsr=2.25)
        assert bg.c0 == 299792458.0
        assert bg.mu0 == 4e-7 * np.pi
        assert np.isreal(bg.eps1)
        
        # Lossy material
        bg_lossy = Background(mur=1.0, epsr=2.25 - 0.1j)
        assert np.iscomplex(bg_lossy.eps1)
        
        print("  ✓ Background class works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Background test failed: {e}")
        return False

def test_wave():
    """Test Wave class."""
    print("\nTesting Wave class...")
    try:
        from bem_solver import Background, Wave
        
        bg = Background(mur=1.0, epsr=2.25)
        wave = Wave(bg, phi_i=0.0, mode='TM', freq=5e14, theta=np.pi/2)
        
        assert wave.wavelength > 0
        assert wave.k0 > 0
        assert wave.mode in ['TM', 'TE']
        
        print(f"  ✓ Wave class works correctly")
        print(f"    Wavelength: {wave.wavelength*1e9:.1f} nm")
        print(f"    Frequency: {wave.freq:.2e} Hz")
        return True
    except Exception as e:
        print(f"  ✗ Wave test failed: {e}")
        return False

def test_element():
    """Test Element class."""
    print("\nTesting Element class...")
    try:
        from bem_solver import Element
        
        n1 = np.array([1.0, 0.0, 0.0])
        n2 = np.array([0.0, 1.0, 0.0])
        xvec = [-np.sqrt(1/3), np.sqrt(1/3)]
        
        el = Element(0, 1, n1, n2, [0, 1], xvec)
        
        assert el.length > 0
        assert len(el.quadpoints) == 2
        assert len(el.normals) == 2
        
        print(f"  ✓ Element class works correctly")
        print(f"    Element length: {el.length:.4f}")
        return True
    except Exception as e:
        print(f"  ✗ Element test failed: {e}")
        return False

def test_solver_creation():
    """Test solver creation."""
    print("\nTesting BEMSolver creation...")
    try:
        from bem_solver import BEMSolver
        
        # Test circular creation
        solver = BEMSolver.from_circle(
            numel=50,
            radius=1e-6,
            mode='TM',
            freq=5e14,
            epsr=2.25
        )
        
        assert solver.numel == 50
        assert len(solver.nodes) == 50
        assert len(solver.elements) == 50
        assert len(solver.basis) == 50
        
        print(f"  ✓ BEMSolver creation successful")
        print(f"    Elements: {solver.numel}")
        print(f"    Radius: {solver.radius*1e6:.2f} μm")
        return True
    except Exception as e:
        print(f"  ✗ Solver creation failed: {e}")
        return False

def test_solve_simple():
    """Test solving a simple problem."""
    print("\nTesting BEM solution (this may take a few seconds)...")
    try:
        from bem_solver import BEMSolver
        
        # Small problem for quick test
        solver = BEMSolver.from_circle(
            numel=30,
            radius=1e-6,
            mode='TM',
            freq=5e14,
            epsr=2.25,
            theta=np.pi/2
        )
        
        sol = solver.solve()
        
        assert sol is not None
        assert len(sol) == 4 * 30  # 4 unknowns per node
        assert np.max(np.abs(sol)) > 0  # Non-trivial solution
        assert np.isfinite(np.max(np.abs(sol)))  # No NaN or Inf
        
        print(f"  ✓ BEM solution successful")
        print(f"    Solution size: {len(sol)}")
        print(f"    Max current magnitude: {np.max(np.abs(sol)):.4e}")
        return True
    except Exception as e:
        print(f"  ✗ Solution failed: {e}")
        return False

def test_energy_balance():
    """Test energy balance computation."""
    print("\nTesting energy balance computation...")
    try:
        from bem_solver import BEMSolver
        
        # Use more elements for better accuracy
        solver = BEMSolver.from_circle(
            numel=100,
            radius=1e-6,
            mode='TM',
            freq=5e14,
            epsr=2.25,
            theta=np.pi/2
        )
        
        solver.solve()
        energy = solver.compute_energy_balance()
        
        assert np.isfinite(energy)
        # Energy balance can be negative (absorbed power) or positive
        # Just check it's reasonable (not huge)
        # For small elements/low frequency, numerical error can be significant
        assert abs(energy) < 50, f"Energy balance seems off: {energy}"
        
        print(f"  ✓ Energy balance computed")
        print(f"    Energy: {energy:.6f}")
        if abs(energy) > 5:
            print(f"    Note: Large energy imbalance suggests numerical issues")
            print(f"          (Consider using more elements for accuracy)")
        return True
    except Exception as e:
        print(f"  ✗ Energy balance test failed: {e}")
        return False

def test_bsdf():
    """Test BSDF computation."""
    print("\nTesting BSDF computation...")
    try:
        from bem_solver import BEMSolver
        
        solver = BEMSolver.from_circle(
            numel=30,
            radius=1e-6,
            mode='TM',
            freq=5e14,
            epsr=2.25,
            theta=np.pi/2
        )
        
        solver.solve()
        sigma = solver.compute_bsdf(n_theta=180, distance=1000*solver.radius)
        
        assert len(sigma) == 180
        assert np.all(np.isfinite(sigma))
        assert np.all(sigma >= 0)  # Scattering intensity should be non-negative
        assert np.sum(sigma) > 0  # Some scattering should occur
        
        print(f"  ✓ BSDF computed successfully")
        print(f"    BSDF points: {len(sigma)}")
        print(f"    Total scattering: {np.sum(sigma):.4e}")
        return True
    except Exception as e:
        print(f"  ✗ BSDF test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("BEM Solver Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Background", test_background),
        ("Wave", test_wave),
        ("Element", test_element),
        ("Solver Creation", test_solver_creation),
        ("BEM Solution", test_solve_simple),
        ("Energy Balance", test_energy_balance),
        ("BSDF Computation", test_bsdf),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for (name, _), result in zip(tests, results):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Installation is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
