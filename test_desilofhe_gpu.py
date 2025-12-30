#!/usr/bin/env python3
"""
Minimal DESILO FHE GPU Test
===========================

This script tests desilofhe GPU mode directly,
matching PrivateInference.py initialization pattern.

Usage:
    python test_desilofhe_gpu.py
"""

import sys
import subprocess


def check_installed_packages():
    """Check which desilofhe packages are installed."""
    print("\n=== Installed desilofhe Packages ===")
    
    # Check pip list for desilofhe packages
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        packages = []
        for line in result.stdout.split('\n'):
            if 'desilofhe' in line.lower():
                packages.append(line.strip())
        
        if packages:
            print("  Found desilofhe packages:")
            for pkg in packages:
                print(f"    {pkg}")
        else:
            print("  [WARN] No desilofhe packages found!")
            
        # Check if both desilofhe and desilofhe-cu* are installed
        has_base = any('desilofhe ' in pkg or pkg.startswith('desilofhe ') for pkg in packages)
        has_cuda = any('desilofhe-cu' in pkg for pkg in packages)
        
        if has_base and has_cuda:
            print("")
            print("  [WARNING] Both base desilofhe and CUDA version are installed!")
            print("  This can cause import conflicts.")
            print("  Solution: pip uninstall desilofhe")
            print("")
            
        return packages
        
    except Exception as e:
        print(f"  Error checking packages: {e}")
        return []


def check_import_location():
    """Check which desilofhe is actually being imported."""
    print("\n=== desilofhe Import Location ===")
    
    try:
        import desilofhe
        
        # Get file location
        file_path = getattr(desilofhe, '__file__', 'Unknown')
        version = getattr(desilofhe, '__version__', 'Unknown')
        
        print(f"  Module file: {file_path}")
        print(f"  Version: {version}")
        
        # Check if it looks like CUDA version
        if 'cu' in file_path.lower() or 'cuda' in file_path.lower():
            print("  [OK] Appears to be CUDA-enabled version")
            return True
        else:
            print("  [WARN] May be CPU-only version")
            print("         GPU mode may not work")
            return False
            
    except ImportError as e:
        print(f"  [FAIL] Cannot import desilofhe: {e}")
        return False


def test_direct_gpu():
    """Test desilofhe GPU mode directly (minimal)."""
    print("\n=== Test 1: Direct GPU mode (minimal) ===")
    try:
        import desilofhe
        print(f"  desilofhe version: {getattr(desilofhe, '__version__', 'unknown')}")
        
        # Minimal GPU test - exactly like DESILO docs
        engine = desilofhe.Engine(mode='gpu')
        print(f"  [OK] Engine(mode='gpu') succeeded!")
        print(f"       Slot count: {engine.slot_count}")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_privateinference_style():
    """Test desilofhe GPU mode like PrivateInference.py."""
    print("\n=== Test 2: PrivateInference.py style ===")
    try:
        import desilofhe
        
        # Exactly like PrivateInference.py _init_desilo()
        mode_str = 'gpu'
        engine_kwargs = {'mode': mode_str}
        engine_kwargs['thread_count'] = 512
        
        print(f"  engine_kwargs: {engine_kwargs}")
        
        engine = desilofhe.Engine(**engine_kwargs)
        print(f"  [OK] Engine(**{engine_kwargs}) succeeded!")
        print(f"       Slot count: {engine.slot_count}")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_with_bootstrap():
    """Test desilofhe GPU mode with bootstrap."""
    print("\n=== Test 3: GPU + Bootstrap ===")
    try:
        import desilofhe
        
        engine_kwargs = {
            'mode': 'gpu',
            'use_bootstrap': True,
            'thread_count': 512
        }
        
        print(f"  engine_kwargs: {engine_kwargs}")
        
        engine = desilofhe.Engine(**engine_kwargs)
        print(f"  [OK] Engine with bootstrap succeeded!")
        print(f"       Slot count: {engine.slot_count}")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def test_cpu_mode():
    """Test desilofhe CPU mode (for comparison)."""
    print("\n=== Test 4: CPU mode (baseline) ===")
    try:
        import desilofhe
        
        # Test 4a: No mode parameter
        print("  4a: Engine() (no mode parameter)")
        try:
            engine = desilofhe.Engine()
            print(f"      [OK] Default engine works, slot_count: {engine.slot_count}")
        except Exception as e:
            print(f"      [FAIL] {e}")
        
        # Test 4b: mode='cpu' explicitly
        print("  4b: Engine(mode='cpu')")
        try:
            engine = desilofhe.Engine(mode='cpu')
            print(f"      [OK] mode='cpu' works, slot_count: {engine.slot_count}")
        except Exception as e:
            print(f"      [FAIL] {e}")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_parallel_mode():
    """Test desilofhe parallel mode."""
    print("\n=== Test 5: Parallel mode ===")
    try:
        import desilofhe
        
        engine_kwargs = {
            'mode': 'parallel',
            'thread_count': 4
        }
        
        print(f"  engine_kwargs: {engine_kwargs}")
        
        engine = desilofhe.Engine(**engine_kwargs)
        print(f"  [OK] Parallel mode works!")
        print(f"       Slot count: {engine.slot_count}")
        return True
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def check_cuda_env():
    """Check CUDA environment variables."""
    print("\n=== CUDA Environment ===")
    import os
    
    env_vars = ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        if var == 'LD_LIBRARY_PATH' and value != 'Not set':
            value = value[:80] + '...' if len(value) > 80 else value
        print(f"  {var}: {value}")


def check_pytorch_cuda():
    """Check PyTorch CUDA status."""
    print("\n=== PyTorch CUDA ===")
    try:
        import torch
        print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("  PyTorch not installed")


def main():
    print("=" * 60)
    print("  DESILO FHE GPU Direct Test")
    print("  (Matching PrivateInference.py initialization)")
    print("=" * 60)
    print(f"\n  Python: {sys.version.split()[0]}")
    
    # First check package installation
    packages = check_installed_packages()
    is_cuda_import = check_import_location()
    
    check_cuda_env()
    check_pytorch_cuda()
    
    results = {}
    results['cpu'] = test_cpu_mode()
    results['parallel'] = test_parallel_mode()
    results['gpu_direct'] = test_direct_gpu()
    results['gpu_pi_style'] = test_privateinference_style()
    results['gpu_bootstrap'] = test_with_bootstrap()
    
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for test, passed in results.items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {test}")
    
    print("\n" + "-" * 60)
    
    # Diagnosis
    if not is_cuda_import:
        print("  [!] You may be importing the wrong desilofhe package!")
        print("      The CPU-only desilofhe is being used instead of desilofhe-cu130")
        print("")
        print("  Solution:")
        print("      pip uninstall desilofhe")
        print("      pip install desilofhe-cu130")
        print("")
    elif results['gpu_direct'] or results['gpu_pi_style']:
        print("  GPU mode is supported!")
        print("  If quickstart.py fails, the issue is in pqc_fhe_integration.py")
    else:
        print("  GPU mode is NOT working in desilofhe")
        print("  Check desilofhe-cu130 installation and CUDA setup")
    print("")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
