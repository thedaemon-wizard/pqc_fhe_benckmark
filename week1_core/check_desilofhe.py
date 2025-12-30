#!/usr/bin/env python3
"""
DESILO FHE Quick Diagnostic
===========================

Run this script to diagnose desilofhe-cu130 GPU issues.
Execute in the SAME virtual environment as PrivateInference.py

Usage:
    python check_desilofhe.py
"""

import sys
import subprocess

def main():
    print("\n" + "=" * 60)
    print("  DESILO FHE Diagnostic")
    print("=" * 60)
    
    # 1. Check installed packages
    print("\n[1] Installed DESILO packages:")
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'list'],
        capture_output=True, text=True
    )
    for line in result.stdout.split('\n'):
        if 'desilo' in line.lower() or 'fhe' in line.lower():
            print(f"    {line}")
    
    # 2. Check import path
    print("\n[2] Import path check:")
    try:
        import desilofhe
        print(f"    desilofhe imported from: {desilofhe.__file__}")
        version = getattr(desilofhe, '__version__', 'N/A')
        print(f"    Version: {version}")
    except ImportError as e:
        print(f"    ERROR: Cannot import desilofhe: {e}")
        return
    
    # 3. Check Engine class
    print("\n[3] Engine class inspection:")
    from desilofhe import Engine
    print(f"    Engine class: {Engine}")
    
    # 4. Test default Engine (no mode parameter)
    print("\n[4] Testing Engine() - default:")
    try:
        engine = Engine()
        print(f"    OK: Default engine created")
        print(f"    slot_count: {getattr(engine, 'slot_count', 'N/A')}")
        del engine
    except Exception as e:
        print(f"    ERROR: {e}")
    
    # 5. Test Engine with mode='gpu'
    print("\n[5] Testing Engine(mode='gpu'):")
    try:
        engine = Engine(mode='gpu')
        print(f"    OK: GPU engine created!")
        print(f"    slot_count: {getattr(engine, 'slot_count', 'N/A')}")
        del engine
    except Exception as e:
        print(f"    ERROR: {e}")
    
    # 6. Test Engine with mode='parallel'
    print("\n[6] Testing Engine(mode='parallel'):")
    try:
        engine = Engine(mode='parallel')
        print(f"    OK: Parallel engine created")
        print(f"    slot_count: {getattr(engine, 'slot_count', 'N/A')}")
        del engine
    except Exception as e:
        print(f"    ERROR: {e}")
    
    # 7. Test Engine with mode='cpu' (PrivateInference.py style)
    print("\n[7] Testing Engine(mode='cpu') - PrivateInference.py style:")
    try:
        engine = Engine(mode='cpu')
        print(f"    OK: CPU engine created")
        print(f"    slot_count: {getattr(engine, 'slot_count', 'N/A')}")
        del engine
    except Exception as e:
        print(f"    ERROR: {e}")
    
    # 8. Test GPU with max_level (PrivateInference.py pattern)
    print("\n[8] Testing Engine(mode='gpu', use_bootstrap=True):")
    try:
        engine = Engine(mode='gpu', use_bootstrap=True)
        print(f"    OK: GPU + Bootstrap engine created!")
        print(f"    slot_count: {getattr(engine, 'slot_count', 'N/A')}")
        del engine
    except Exception as e:
        print(f"    ERROR: {e}")
    
    # 9. Environment check
    print("\n[9] Environment:")
    import os
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH', 'Not set')
    ld_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
    print(f"    CUDA_HOME: {cuda_home}")
    print(f"    LD_LIBRARY_PATH: {ld_path[:100]}..." if len(ld_path) > 100 else f"    LD_LIBRARY_PATH: {ld_path}")
    print(f"    Python: {sys.executable}")
    print(f"    Working dir: {os.getcwd()}")
    
    # 10. PyTorch CUDA check
    print("\n[10] PyTorch CUDA:")
    try:
        import torch
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    CUDA version: {torch.version.cuda}")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("    PyTorch not installed")
    
    print("\n" + "=" * 60)
    print("  Diagnostic complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
