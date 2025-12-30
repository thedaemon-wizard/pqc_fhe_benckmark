#!/usr/bin/env python3
"""
DESILO FHE GPU Diagnostic Script
================================

This script checks GPU availability and DESILO FHE GPU support.
Run this to diagnose GPU mode issues.

Usage:
    python diagnose_gpu.py
"""

import sys
import subprocess
import os


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(name: str, success: bool, details: str = ""):
    """Print test result."""
    status = "[OK]" if success else "[FAIL]"
    print(f"  {status} {name}")
    if details:
        print(f"       {details}")


def check_nvidia_driver():
    """Check NVIDIA driver installation."""
    print_header("NVIDIA Driver Check")
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                parts = line.split(', ')
                if len(parts) >= 3:
                    print_result(
                        f"GPU {i}", 
                        True, 
                        f"{parts[0]}, Driver: {parts[1]}, Memory: {parts[2]}"
                    )
            return True
        else:
            print_result("nvidia-smi", False, "NVIDIA driver not found")
            return False
    except FileNotFoundError:
        print_result("nvidia-smi", False, "Command not found - NVIDIA driver not installed")
        return False
    except Exception as e:
        print_result("nvidia-smi", False, str(e))
        return False


def check_cuda_toolkit():
    """Check CUDA toolkit installation."""
    print_header("CUDA Toolkit Check")
    
    # Check nvcc
    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Extract version
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print_result("nvcc", True, line.strip())
                    break
            cuda_found = True
        else:
            print_result("nvcc", False, "CUDA compiler not found")
            cuda_found = False
    except FileNotFoundError:
        print_result("nvcc", False, "Command not found")
        cuda_found = False
    except Exception as e:
        print_result("nvcc", False, str(e))
        cuda_found = False
    
    # Check CUDA environment variables
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        print_result("CUDA_HOME", True, cuda_home)
    else:
        print_result("CUDA_HOME", False, "Not set")
    
    # Check LD_LIBRARY_PATH
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if 'cuda' in ld_path.lower():
        print_result("LD_LIBRARY_PATH", True, "Contains CUDA paths")
    else:
        print_result("LD_LIBRARY_PATH", False, "May not include CUDA libs")
    
    return cuda_found


def check_pytorch_cuda():
    """Check PyTorch CUDA support."""
    print_header("PyTorch CUDA Check")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print_result(
            "torch.cuda.is_available()", 
            cuda_available,
            f"CUDA: {cuda_available}"
        )
        
        if cuda_available:
            print_result(
                "CUDA Version",
                True,
                torch.version.cuda or "N/A"
            )
            print_result(
                "GPU Count",
                True,
                str(torch.cuda.device_count())
            )
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print_result(
                    f"GPU {i}",
                    True,
                    f"{props.name}, {props.total_memory // 1024**3}GB"
                )
        
        return cuda_available
        
    except ImportError:
        print_result("PyTorch", False, "Not installed")
        return False
    except Exception as e:
        print_result("PyTorch CUDA", False, str(e))
        return False


def check_desilofhe():
    """Check desilofhe installation and GPU support."""
    print_header("DESILO FHE Check")
    
    try:
        import desilofhe
        
        # Get version if available
        version = getattr(desilofhe, '__version__', 'Unknown')
        print_result("desilofhe", True, f"Version: {version}")
        
        # Try to check available modes
        # Try creating a default engine first
        try:
            engine = desilofhe.Engine()
            print_result("CPU mode", True, "Default engine works")
        except Exception as e:
            print_result("CPU mode", False, str(e))
        
        # Try GPU mode
        try:
            gpu_engine = desilofhe.Engine(mode="gpu")
            print_result("GPU mode", True, "GPU engine works!")
            gpu_slot_count = getattr(gpu_engine, 'slot_count', 'N/A')
            print(f"       GPU engine slot count: {gpu_slot_count}")
            return True
        except RuntimeError as e:
            if "Not supported mode" in str(e):
                print_result("GPU mode", False, "Not compiled with GPU support")
            else:
                print_result("GPU mode", False, str(e))
            return False
        except Exception as e:
            print_result("GPU mode", False, str(e))
            return False
            
    except ImportError:
        print_result("desilofhe", False, "Not installed")
        print("\n  Install with: pip install desilofhe")
        return False


def check_liberate_fhe():
    """Check liberate-fhe installation (GPU version)."""
    print_header("Liberate FHE Check (GPU Version)")
    
    try:
        from liberate import fhe
        from liberate.fhe import presets
        
        print_result("liberate-fhe", True, "Installed")
        
        # Check presets
        available_presets = list(presets.params.keys())
        print_result("Presets", True, ", ".join(available_presets))
        
        # Try creating engine
        try:
            params = presets.params["silver"].copy()
            params["num_scales"] = 5
            engine = fhe.ckks_engine(**params, verbose=False)
            print_result("GPU Engine", True, "Successfully created!")
            return True
        except Exception as e:
            print_result("GPU Engine", False, str(e))
            return False
            
    except ImportError:
        print_result("liberate-fhe", False, "Not installed")
        print("\n  Install with: pip install liberate-fhe")
        print("  Or from source: https://github.com/Desilo/liberate-fhe")
        return False


def main():
    """Run all diagnostic checks."""
    print("\n")
    print("+" + "=" * 58 + "+")
    print("|     DESILO FHE GPU Diagnostic Tool                      |")
    print("+" + "=" * 58 + "+")
    print(f"\n  Python: {sys.version.split()[0]}")
    
    results = {}
    
    # Run all checks
    results['nvidia'] = check_nvidia_driver()
    results['cuda'] = check_cuda_toolkit()
    results['pytorch'] = check_pytorch_cuda()
    results['desilofhe'] = check_desilofhe()
    results['liberate'] = check_liberate_fhe()
    
    # Summary
    print_header("Summary")
    
    print("\n  Hardware/Driver:")
    print_result("NVIDIA Driver", results['nvidia'])
    print_result("CUDA Toolkit", results['cuda'])
    print_result("PyTorch CUDA", results['pytorch'])
    
    print("\n  DESILO FHE:")
    print_result("desilofhe (standard)", results['desilofhe'])
    print_result("liberate-fhe (GPU)", results['liberate'])
    
    # Recommendations
    print_header("Recommendations")
    
    if not results['nvidia']:
        print("  1. Install NVIDIA driver:")
        print("     sudo dnf install nvidia-driver  # Fedora/RHEL")
        print("     sudo apt install nvidia-driver-535  # Ubuntu")
        print("")
    
    if not results['cuda']:
        print("  2. Install CUDA Toolkit:")
        print("     https://developer.nvidia.com/cuda-downloads")
        print("     Set CUDA_HOME and add to PATH/LD_LIBRARY_PATH")
        print("")
    
    if results['nvidia'] and results['cuda']:
        if not results['desilofhe'] and not results['liberate']:
            print("  3. Install DESILO FHE with GPU support:")
            print("")
            print("     Option A - liberate-fhe (recommended):")
            print("       pip install liberate-fhe")
            print("")
            print("     Option B - Build desilofhe from source:")
            print("       git clone https://github.com/Desilo/liberate-fhe.git")
            print("       cd liberate-fhe")
            print("       pip install -e .")
            print("")
        elif results['desilofhe'] and not results['liberate']:
            print("  Your desilofhe doesn't have GPU support.")
            print("  Try installing liberate-fhe for GPU operations:")
            print("    pip install liberate-fhe")
            print("")
    
    if results['liberate'] or (results['desilofhe'] and results['nvidia']):
        print("  GPU support is available!")
        print("  Use: FHEConfig(mode='gpu')")
        print("")
    
    # Final status
    gpu_ready = (results['nvidia'] and results['cuda'] and 
                 (results['liberate'] or results['desilofhe']))
    
    print("-" * 60)
    if gpu_ready:
        print("  Status: GPU mode should be available")
    else:
        print("  Status: GPU mode not available - using CPU fallback")
    print("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
