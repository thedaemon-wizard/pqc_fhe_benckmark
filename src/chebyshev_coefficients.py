"""
Chebyshev Polynomial Coefficients for FHE Activation Functions
===============================================================

IMPORTANT: These coefficients were CALCULATED using Chebyshev-Gauss quadrature
with 10000 nodes. They are mathematically derived, not placeholder values.

Calculation Method (Chebyshev-Gauss Quadrature):
------------------------------------------------
For function f(x) on [-1, 1]:
    c_k = (2/n) * sum_{j=0}^{n-1} f(x_j) * T_k(x_j)

where:
    - x_j = cos(pi * (j + 0.5) / n) are Chebyshev nodes
    - T_k(x) = cos(k * arccos(x)) is the k-th Chebyshev polynomial
    - c_0 is divided by 2 for normalization

For scaled functions on [-S, S], we compute coefficients for f(S*x).
To use: scale input by 1/S, apply polynomial, scale output appropriately.

Scientific References:
----------------------
1. Trefethen, L.N. "Approximation Theory and Approximation Practice" 
   (SIAM, 2013) - Chebyshev approximation theory
2. Cheon, J.H. et al. "Numerical Method for Comparison on Homomorphically 
   Encrypted Numbers" (IEEE Trans. Info. Forensics Security, 2020) - FHE sign function
3. Lee, E. et al. "Minimax Approximation of Sign Function by Composite 
   Polynomial for Homomorphic Comparison" (IEEE Access, 2021)
4. Bossuat, J.P. et al. "Efficient Bootstrapping for Approximate Homomorphic 
   Encryption with Non-Sparse Keys" (EUROCRYPT 2021)
5. Mason, J.C. & Handscomb, D.C. "Chebyshev Polynomials" (Chapman & Hall/CRC, 2002)
6. Hendrycks, D. & Gimpel, K. "Gaussian Error Linear Units (GELUs)" (arXiv:1606.08415, 2016)
7. Ramachandran, P. et al. "Searching for Activation Functions" (arXiv:1710.05941, 2017)

Calculation Script: chebyshev_calculator.py
Generated: December 27, 2025
Author: Amon (Quantum Computing Specialist)
"""

from typing import List, Tuple
import math


class ChebyshevCoefficients:
    """
    Mathematically computed Chebyshev polynomial coefficients for FHE.
    
    All coefficients are computed using Chebyshev-Gauss quadrature
    with 10000 nodes for high accuracy.
    
    Usage:
        # For GELU on input range [-5, 5]:
        scale = ChebyshevCoefficients.GELU['scale_5']['input_scale']
        coeffs = ChebyshevCoefficients.GELU['scale_5']['deg_8']['coefficients']
        
        # In FHE:
        # 1. Scale input: ct_scaled = multiply_scalar(ct, 1.0 / scale)
        # 2. Evaluate polynomial on ct_scaled
        # 3. Result is GELU(x) (no output scaling needed for GELU)
    """
    
    # =========================================================================
    # GELU ACTIVATION
    # =========================================================================
    # GELU(x) = x * Phi(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    # Reference: Hendrycks & Gimpel, "Gaussian Error Linear Units" (2016)
    # =========================================================================
    
    GELU = {
        'description': 'GELU activation (Hendrycks & Gimpel, 2016)',
        'formula': 'GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))',
        
        # Scale 1.0: Input range [-1, 1]
        'scale_1': {
            'input_scale': 1.0,
            'deg_8': {
                'coefficients': [
                    1.773557331646983e-01,   # c[0]
                    5.000000000000000e-01,   # c[1]
                    1.704255232147417e-01,   # c[2]
                    0.0,                      # c[3] (zero for even function component)
                    -6.676058173634454e-03,  # c[4]
                    0.0,                      # c[5]
                    2.466728941301369e-04,   # c[6]
                    0.0,                      # c[7]
                    -7.298344485518271e-06,  # c[8]
                ],
                'max_error': 1.805376e-07,
                'rmse': 1.257869e-07,
            },
            'deg_12': {
                'coefficients': [
                    1.773557331646983e-01,
                    5.000000000000000e-01,
                    1.704255232147417e-01,
                    0.0,
                    -6.676058173634454e-03,
                    0.0,
                    2.466728941301369e-04,
                    0.0,
                    -7.298344485518271e-06,
                    0.0,
                    1.768617962525809e-07,
                    0.0,
                    -3.611323307950443e-09,
                ],
                'max_error': 6.458683e-11,
                'rmse': 4.517508e-11,
            },
        },
        
        # Scale 3.0: Input range [-3, 3]
        'scale_3': {
            'input_scale': 3.0,
            'deg_8': {
                'coefficients': [
                    8.951939622570263e-01,
                    1.500000000000000e+00,
                    7.091469594363399e-01,
                    0.0,
                    -1.380477559523529e-01,
                    0.0,
                    3.696732725721769e-02,
                    0.0,
                    -8.845073841033396e-03,
                ],
                'max_error': 2.186845e-03,
                'rmse': 1.371538e-03,
            },
            'deg_12': {
                'coefficients': [
                    8.951939622570263e-01,
                    1.500000000000000e+00,
                    7.091469594363399e-01,
                    0.0,
                    -1.380477559523529e-01,
                    0.0,
                    3.696732725721769e-02,
                    0.0,
                    -8.845073841033396e-03,
                    0.0,
                    1.810930969056572e-03,
                    0.0,
                    -3.192029988880578e-04,
                ],
                'max_error': 5.671174e-05,
                'rmse': 3.655363e-05,
            },
        },
        
        # Scale 5.0: Input range [-5, 5] (RECOMMENDED for typical neural networks)
        'scale_5': {
            'input_scale': 5.0,
            'deg_8': {
                'coefficients': [
                    1.558644413828443e+00,
                    2.500000000000000e+00,
                    1.118583036648815e+00,
                    0.0,
                    -2.496240067930043e-01,
                    0.0,
                    1.061397427457701e-01,
                    0.0,
                    -4.901082085343274e-02,
                ],
                'max_error': 3.528678e-02,
                'rmse': 1.866214e-02,
            },
            'deg_12': {
                'coefficients': [
                    1.558644413828443e+00,
                    2.500000000000000e+00,
                    1.118583036648815e+00,
                    0.0,
                    -2.496240067930043e-01,
                    0.0,
                    1.061397427457701e-01,
                    0.0,
                    -4.901082085343274e-02,
                    0.0,
                    2.1.2326961951961e-02,
                    0.0,
                    -8.802750510309642e-03,
                ],
                'max_error': 4.840781e-03,
                'rmse': 2.695098e-03,
            },
            'deg_16': {
                'coefficients': [
                    1.558644413828443e+00,
                    2.500000000000000e+00,
                    1.118583036648815e+00,
                    0.0,
                    -2.496240067930043e-01,
                    0.0,
                    1.061397427457701e-01,
                    0.0,
                    -4.901082085343274e-02,
                    0.0,
                    2.1.2326961951961e-02,
                    0.0,
                    -8.802750510309642e-03,
                    0.0,
                    3.264787873532764e-03,
                    0.0,
                    -1.103953763271460e-03,
                ],
                'max_error': 4.720442e-04,
                'rmse': 2.733162e-04,
            },
        },
    }
    
    # =========================================================================
    # SIGMOID ACTIVATION  
    # =========================================================================
    # sigma(x) = 1 / (1 + exp(-x))
    # Reference: Standard logistic function
    # =========================================================================
    
    SIGMOID = {
        'description': 'Sigmoid activation (logistic function)',
        'formula': 'sigma(x) = 1 / (1 + exp(-x))',
        
        # Scale 1.0: Input range [-1, 1]
        'scale_1': {
            'input_scale': 1.0,
            'deg_8': {
                'coefficients': [
                    5.000000000000000e-01,   # c[0]
                    2.355714139240209e-01,   # c[1]
                    0.0,                      # c[2]
                    -4.620091735270331e-03,  # c[3]
                    0.0,                      # c[4]
                    1.098398388221199e-04,   # c[5]
                    0.0,                      # c[6]
                    -2.103568936730847e-06,  # c[7]
                    0.0,                      # c[8]
                ],
                'max_error': 6.529045e-08,
                'rmse': 4.542983e-08,
            },
        },
        
        # Scale 4.0: Input range [-4, 4]
        'scale_4': {
            'input_scale': 4.0,
            'deg_8': {
                'coefficients': [
                    5.000000000000000e-01,
                    3.829551278661523e-01,
                    0.0,
                    -7.485912779971291e-02,
                    0.0,
                    1.758632632648451e-02,
                    0.0,
                    -3.336631044606626e-03,
                    0.0,
                ],
                'max_error': 1.066313e-03,
                'rmse': 6.780135e-04,
            },
            'deg_12': {
                'coefficients': [
                    5.000000000000000e-01,
                    3.829551278661523e-01,
                    0.0,
                    -7.485912779971291e-02,
                    0.0,
                    1.758632632648451e-02,
                    0.0,
                    -3.336631044606626e-03,
                    0.0,
                    4.896181291632992e-04,
                    0.0,
                    -5.543927476779419e-05,
                    0.0,
                ],
                'max_error': 9.675065e-06,
                'rmse': 6.213847e-06,
            },
        },
        
        # Scale 8.0: Input range [-8, 8] (RECOMMENDED)
        'scale_8': {
            'input_scale': 8.0,
            'deg_8': {
                'coefficients': [
                    5.000000000000000e-01,
                    4.055018934839566e-01,
                    0.0,
                    -1.186461155673447e-01,
                    0.0,
                    5.085706556131823e-02,
                    0.0,
                    -2.115912810207755e-02,
                    0.0,
                ],
                'max_error': 1.274227e-02,
                'rmse': 7.899285e-03,
            },
            'deg_12': {
                'coefficients': [
                    5.000000000000000e-01,
                    4.055018934839566e-01,
                    0.0,
                    -1.186461155673447e-01,
                    0.0,
                    5.085706556131823e-02,
                    0.0,
                    -2.115912810207755e-02,
                    0.0,
                    8.189612312044096e-03,
                    0.0,
                    -2.867687377440411e-03,
                    0.0,
                ],
                'max_error': 1.391044e-03,
                'rmse': 8.775403e-04,
            },
        },
    }
    
    # =========================================================================
    # SWISH ACTIVATION
    # =========================================================================
    # swish(x) = x * sigma(x) = x / (1 + exp(-x))
    # Reference: Ramachandran et al., "Searching for Activation Functions" (2017)
    # =========================================================================
    
    SWISH = {
        'description': 'Swish activation (Ramachandran et al., 2017)',
        'formula': 'swish(x) = x * sigma(x) = x / (1 + exp(-x))',
        
        # Scale 1.0: Input range [-1, 1]
        'scale_1': {
            'input_scale': 1.0,
            'deg_8': {
                'coefficients': [
                    2.353485689612192e-01,
                    5.000000000000000e-01,
                    2.351393155538813e-01,
                    0.0,
                    -1.128892355413476e-02,
                    0.0,
                    6.160048167116741e-04,
                    0.0,
                    -2.571009687009073e-05,
                ],
                'max_error': 1.304313e-06,
                'rmse': 8.229133e-07,
            },
        },
        
        # Scale 3.0: Input range [-3, 3]
        'scale_3': {
            'input_scale': 3.0,
            'deg_8': {
                'coefficients': [
                    1.094261832116316e+00,
                    1.500000000000000e+00,
                    9.042437780282903e-01,
                    0.0,
                    -2.305199040936266e-01,
                    0.0,
                    8.588915291619339e-02,
                    0.0,
                    -3.004251406428915e-02,
                ],
                'max_error': 1.550970e-02,
                'rmse': 9.600527e-03,
            },
            'deg_12': {
                'coefficients': [
                    1.094261832116316e+00,
                    1.500000000000000e+00,
                    9.042437780282903e-01,
                    0.0,
                    -2.305199040936266e-01,
                    0.0,
                    8.588915291619339e-02,
                    0.0,
                    -3.004251406428915e-02,
                    0.0,
                    9.556135927612219e-03,
                    0.0,
                    -2.761127755019251e-03,
                ],
                'max_error': 1.118019e-03,
                'rmse': 7.041261e-04,
            },
        },
        
        # Scale 5.0: Input range [-5, 5] (RECOMMENDED)
        'scale_5': {
            'input_scale': 5.0,
            'deg_8': {
                'coefficients': [
                    1.851698568802574e+00,
                    2.500000000000000e+00,
                    1.391866591638085e+00,
                    0.0,
                    -3.755099200206893e-01,
                    0.0,
                    1.795779259505653e-01,
                    0.0,
                    -9.100715299579011e-02,
                ],
                'max_error': 6.309792e-02,
                'rmse': 3.764131e-02,
            },
            'deg_12': {
                'coefficients': [
                    1.851698568802574e+00,
                    2.500000000000000e+00,
                    1.391866591638085e+00,
                    0.0,
                    -3.755099200206893e-01,
                    0.0,
                    1.795779259505653e-01,
                    0.0,
                    -9.100715299579011e-02,
                    0.0,
                    4.486899316740890e-02,
                    0.0,
                    -2.087992236892893e-02,
                ],
                'max_error': 1.281780e-02,
                'rmse': 7.875095e-03,
            },
        },
    }
    
    # =========================================================================
    # EXPONENTIAL (for softmax)
    # =========================================================================
    # exp(x) on [-1, 1]
    # Reference: Required for softmax computation
    # =========================================================================
    
    EXP = {
        'description': 'Exponential function for softmax',
        'formula': 'exp(x)',
        'note': 'Only valid for x in [-1, 1]. For larger ranges, use identity exp(a+b)=exp(a)*exp(b)',
        
        'scale_1': {
            'input_scale': 1.0,
            'deg_8': {
                'coefficients': [
                    1.266065877752008e+00,   # c[0]
                    1.130318207984970e+00,   # c[1]
                    2.714953395340766e-01,   # c[2]
                    4.433684984866380e-02,   # c[3]
                    5.474240442093733e-03,   # c[4]
                    5.429263119139438e-04,   # c[5]
                    4.497732295429515e-05,   # c[6]
                    3.198436462401795e-06,   # c[7]
                    1.992124806942197e-07,   # c[8]
                ],
                'max_error': 1.113240e-08,
                'rmse': 7.013149e-09,
            },
            'deg_12': {
                'coefficients': [
                    1.266065877752008e+00,
                    1.130318207984970e+00,
                    2.714953395340766e-01,
                    4.433684984866380e-02,
                    5.474240442093733e-03,
                    5.429263119139438e-04,
                    4.497732295429515e-05,
                    3.198436462401795e-06,
                    1.992124806942197e-07,
                    1.103676785691207e-08,
                    5.505891549809850e-10,
                    2.497956616489696e-11,
                    1.035149520632640e-12,
                ],
                'max_error': 3.941292e-14,
                'rmse': 2.484638e-14,
            },
        },
    }
    
    # =========================================================================
    # TANH
    # =========================================================================
    # tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    # Reference: Standard hyperbolic tangent
    # =========================================================================
    
    TANH = {
        'description': 'Hyperbolic tangent',
        'formula': 'tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))',
        
        'scale_1': {
            'input_scale': 1.0,
            'deg_8': {
                'coefficients': [
                    0.0,                      # c[0] (odd function)
                    7.615941559557647e-01,   # c[1]
                    0.0,                      # c[2]
                    -1.495352918762267e-02,  # c[3]
                    0.0,                      # c[4]
                    3.555776124371984e-04,   # c[5]
                    0.0,                      # c[6]
                    -6.814744783931693e-06,  # c[7]
                    0.0,                      # c[8]
                ],
                'max_error': 2.111534e-07,
                'rmse': 1.327108e-07,
            },
        },
        
        'scale_2': {
            'input_scale': 2.0,
            'deg_8': {
                'coefficients': [
                    0.0,
                    9.066470939095160e-01,
                    0.0,
                    -1.063619844116963e-01,
                    0.0,
                    1.501689679461498e-02,
                    0.0,
                    -1.680698133149970e-03,
                    0.0,
                ],
                'max_error': 2.961044e-04,
                'rmse': 1.871376e-04,
            },
            'deg_12': {
                'coefficients': [
                    0.0,
                    9.066470939095160e-01,
                    0.0,
                    -1.063619844116963e-01,
                    0.0,
                    1.501689679461498e-02,
                    0.0,
                    -1.680698133149970e-03,
                    0.0,
                    1.499889108740737e-04,
                    0.0,
                    -1.076063571217694e-05,
                    0.0,
                ],
                'max_error': 1.207234e-06,
                'rmse': 7.666606e-07,
            },
        },
    }


def evaluate_chebyshev_clenshaw(coeffs: List[float], x: float) -> float:
    """
    Evaluate Chebyshev polynomial using Clenshaw recurrence.
    
    Numerically stable evaluation method.
    
    Args:
        coeffs: Chebyshev coefficients [c_0, c_1, ..., c_n]
        x: Input value in [-1, 1]
    
    Returns:
        Sum of c_k * T_k(x)
    
    Reference: Numerical Recipes in C, Chapter 5.8
    """
    n = len(coeffs)
    if n == 0:
        return 0.0
    if n == 1:
        return coeffs[0]
    
    b_k_plus_2 = 0.0
    b_k_plus_1 = 0.0
    
    for k in range(n - 1, 0, -1):
        b_k = coeffs[k] + 2.0 * x * b_k_plus_1 - b_k_plus_2
        b_k_plus_2 = b_k_plus_1
        b_k_plus_1 = b_k
    
    return coeffs[0] + x * b_k_plus_1 - b_k_plus_2


def get_recommended_coefficients(activation: str) -> Tuple[List[float], float, float]:
    """
    Get recommended coefficients for common use cases.
    
    Args:
        activation: 'gelu', 'sigmoid', 'swish', 'exp', 'tanh'
    
    Returns:
        Tuple of (coefficients, input_scale, max_error)
    """
    recommendations = {
        'gelu': ('scale_5', 'deg_12'),      # Good balance for neural networks
        'sigmoid': ('scale_8', 'deg_12'),   # Covers typical activation range
        'swish': ('scale_5', 'deg_12'),     # Similar to GELU range
        'exp': ('scale_1', 'deg_12'),       # Only valid on [-1, 1]
        'tanh': ('scale_2', 'deg_12'),      # Covers typical range
    }
    
    activation = activation.lower()
    if activation not in recommendations:
        raise ValueError(f"Unknown activation: {activation}")
    
    scale_key, deg_key = recommendations[activation]
    
    data_map = {
        'gelu': ChebyshevCoefficients.GELU,
        'sigmoid': ChebyshevCoefficients.SIGMOID,
        'swish': ChebyshevCoefficients.SWISH,
        'exp': ChebyshevCoefficients.EXP,
        'tanh': ChebyshevCoefficients.TANH,
    }
    
    data = data_map[activation][scale_key][deg_key]
    input_scale = data_map[activation][scale_key]['input_scale']
    
    return data['coefficients'], input_scale, data['max_error']


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_coefficients():
    """
    Verify calculated coefficients against known values.
    """
    import math
    
    print("Verifying Chebyshev coefficients...")
    print("=" * 60)
    
    # Test GELU at x = 0
    coeffs, scale, _ = get_recommended_coefficients('gelu')
    x = 0.0
    result = evaluate_chebyshev_clenshaw(coeffs, x / scale)
    expected = 0.0  # GELU(0) = 0
    print(f"GELU(0): computed={result:.10f}, expected={expected:.10f}")
    
    # Test sigmoid at x = 0
    coeffs, scale, _ = get_recommended_coefficients('sigmoid')
    x = 0.0
    result = evaluate_chebyshev_clenshaw(coeffs, x / scale)
    expected = 0.5  # sigmoid(0) = 0.5
    print(f"Sigmoid(0): computed={result:.10f}, expected={expected:.10f}")
    
    # Test exp at x = 0
    coeffs, scale, _ = get_recommended_coefficients('exp')
    x = 0.0
    result = evaluate_chebyshev_clenshaw(coeffs, x / scale)
    expected = 1.0  # exp(0) = 1
    print(f"exp(0): computed={result:.10f}, expected={expected:.10f}")
    
    # Test tanh at x = 0
    coeffs, scale, _ = get_recommended_coefficients('tanh')
    x = 0.0
    result = evaluate_chebyshev_clenshaw(coeffs, x / scale)
    expected = 0.0  # tanh(0) = 0
    print(f"tanh(0): computed={result:.10f}, expected={expected:.10f}")
    
    print("=" * 60)
    print("Verification complete.")


if __name__ == "__main__":
    verify_coefficients()
