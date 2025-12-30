"""
DESILO FHE Engine - DESILO API Compliant Implementation v2.1.0
================================================================

Chebyshev Coefficient Computation:
----------------------------------
All coefficients are COMPUTED using NumPy's Chebyshev polynomial fitting:
    coeffs = numpy.polynomial.chebyshev.chebfit(x, y, degree)

where x is Chebyshev nodes and y is the function values at those nodes.

References:
----------
- DESILO FHE Documentation: https://fhe.desilo.dev/latest/
- Hendrycks, D., Gimpel, K. (2016). "Gaussian Error Linear Units (GELUs)"
  https://arxiv.org/abs/1606.08415
- Ramachandran, P., et al. (2017). "Searching for Activation Functions"
  https://arxiv.org/abs/1710.05941
- Lee, J., et al. (2022). "Privacy-Preserving Machine Learning with FHE"
  https://arxiv.org/abs/2106.07229
- Cheon, J.H., et al. (2018). "A Full RNS Variant of Approximate HE"
  https://eprint.iacr.org/2018/931
- Press, W.H., et al. "Numerical Recipes", Chapter 5.8: Chebyshev Approximation
- Mason, J.C., Handscomb, D.C. (2003). "Chebyshev Polynomials"
- PrivateInference.py v5.6.1

Author: Amon (Quantum Computing Specialist)
License: MIT
Version: 2.1.0
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Any, Dict, Tuple, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# CHEBYSHEV POLYNOMIAL COEFFICIENTS - SCIENTIFICALLY COMPUTED
# =============================================================================

class ChebyshevActivations:
    """
    Chebyshev polynomial approximations for activation functions
    
    =========================================================================
    COMPUTATION METHOD:
    =========================================================================
    All coefficients are COMPUTED using NumPy's Chebyshev polynomial fitting:
    
        >>> from numpy.polynomial import chebyshev as cheb
        >>> x_cheb = np.cos(np.pi * (2*k + 1) / (2*n))  # Chebyshev nodes
        >>> y = func(scale * x_cheb)                    # Function values
        >>> coeffs = cheb.chebfit(x_cheb, y, degree)    # Fit polynomial
    
    These approximations are valid for inputs in [-1, 1] range.
    For larger ranges, scale input first, then scale output back.
    
    POLYNOMIAL DEGREE: 8
    
    REFERENCES:
        - NumPy polynomial.chebyshev documentation
        - Press et al., "Numerical Recipes", Chapter 5.8
        - Mason & Handscomb (2003), "Chebyshev Polynomials"
        - DESILO FHE Advanced_Examples.pdf
        - PrivateInference.py v5.6.1 ChebyshevActivationsV56
    """
    
    # =========================================================================
    # GELU (Gaussian Error Linear Unit)
    # =========================================================================
    # Formula: GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    # Computed using numpy.polynomial.chebyshev.chebfit
    # Degree: 8, Scale: 3.0, Max Error: 2.187e-03
    # Reference: Hendrycks & Gimpel (2016), arXiv:1606.08415
    # =========================================================================
    GELU = [
        8.9519396225702619e-01,   # c0
        1.5000000000000002e+00,   # c1
        7.0914695943634010e-01,   # c2
        9.1208170398770153e-17,   # c3 (effectively 0)
        -1.3804775595235266e-01,  # c4
        7.8199406966373600e-18,   # c5 (effectively 0)
        3.6967327257217367e-02,   # c6
        2.8268213511875512e-16,   # c7 (effectively 0)
        -8.8450738410338538e-03,  # c8
    ]
    GELU_SCALE = 3.0
    GELU_MAX_ERROR = 2.187e-03
    
    # =========================================================================
    # SIGMOID (Logistic Function)
    # =========================================================================
    # Formula: sigma(x) = 1 / (1 + exp(-x))
    # Computed using numpy.polynomial.chebyshev.chebfit
    # Degree: 8, Scale: 3.0, Max Error: 2.890e-04
    # Reference: Kim et al. (2018), doi:10.1186/s12920-018-0401-7
    # =========================================================================
    SIGMOID = [
        5.0000000000000033e-01,   # c0
        5.0543582494364436e-01,   # c1
        -1.7448985751401349e-17,  # c2 (effectively 0)
        -6.1102911134640864e-02,  # c3
        6.0384613284239415e-17,   # c4 (effectively 0)
        9.5597061005403237e-03,   # c5
        -2.1170821708829100e-16,  # c6 (effectively 0)
        -1.5301492875703941e-03,  # c7
        -1.9443027365464061e-16,  # c8 (effectively 0)
    ]
    SIGMOID_SCALE = 3.0
    SIGMOID_MAX_ERROR = 2.890e-04
    
    # =========================================================================
    # SWISH (Self-Gated Activation / SiLU)
    # =========================================================================
    # Formula: swish(x) = x * sigma(x) = x / (1 + exp(-x))
    # Computed using numpy.polynomial.chebyshev.chebfit
    # Degree: 8, Scale: 3.0, Max Error: 3.685e-04
    # Reference: Ramachandran et al. (2017), arXiv:1710.05941
    # =========================================================================
    SWISH = [
        7.5815373741546632e-01,   # c0
        1.5000000000000011e+00,   # c1
        6.6649937071350496e-01,   # c2
        2.7688022145277653e-16,   # c3 (effectively 0)
        -7.7314807551150694e-02,  # c4
        2.5754971244764713e-18,   # c5 (effectively 0)
        1.2044335219454121e-02,   # c6
        3.6610298250911758e-16,   # c7 (effectively 0)
        -1.9267495577120319e-03,  # c8
    ]
    SWISH_SCALE = 3.0
    SWISH_MAX_ERROR = 3.685e-04
    
    # =========================================================================
    # TANH (Hyperbolic Tangent)
    # =========================================================================
    # Formula: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    # Computed using numpy.polynomial.chebyshev.chebfit
    # Degree: 8, Scale: 3.0, Max Error: 1.950e-02
    # =========================================================================
    TANH = [
        3.6637359812630174e-17,   # c0 (effectively 0, odd function)
        1.2073855733208407e+00,   # c1
        -1.0415550083712476e-16,  # c2 (effectively 0)
        -2.8401846109749956e-01,  # c3
        5.0307679358860494e-17,   # c4 (effectively 0)
        9.7513958889868704e-02,   # c5
        -3.4219480282556084e-16,  # c6 (effectively 0)
        -3.5250937832798750e-02,  # c7
        -2.9584916239279461e-16,  # c8 (effectively 0)
    ]
    TANH_SCALE = 3.0
    TANH_MAX_ERROR = 1.950e-02
    
    # =========================================================================
    # EXP (Exponential, for Softmax)
    # =========================================================================
    # Formula: exp(x)
    # Computed using numpy.polynomial.chebyshev.chebfit
    # Degree: 8, Scale: 1.0, Max Error: 1.161e-08
    # Reference: Cheon et al. (2018), ePrint 2018/931
    # =========================================================================
    EXP = [
        1.2660658777520082e+00,   # c0
        1.1303182079849701e+00,   # c1
        2.7149533953407656e-01,   # c2
        4.4336849848663776e-02,   # c3
        5.4742404420938225e-03,   # c4
        5.4292631191396135e-04,   # c5
        4.4977322954076628e-05,   # c6
        3.1984364625917805e-06,   # c7
        1.9921248036519700e-07,   # c8
    ]
    EXP_SCALE = 1.0
    EXP_MAX_ERROR = 1.161e-08
    
    # =========================================================================
    # RELU_SMOOTH (Smooth ReLU Approximation)
    # =========================================================================
    # Formula: relu_smooth(x) = 0.5 * x * (1 + tanh(alpha * x))
    # Computed using numpy.polynomial.chebyshev.chebfit
    # Degree: 8, Scale: 5.0, Max Error: 3.997e-15 (machine precision)
    # Reference: Glorot et al. (2011), "Deep Sparse Rectifier Neural Networks"
    # =========================================================================
    RELU_SMOOTH = [
        6.2460970023287550e-02,   # c0
        2.5000000000000018e+00,   # c1
        6.2447965448719819e-02,   # c2
        3.8124836955125788e-16,   # c3 (effectively 0)
        -1.3001325115775288e-05,  # c4
        8.9566992165430217e-17,   # c5 (effectively 0)
        3.2486296026533738e-09,   # c6
        6.5842037363114449e-16,   # c7 (effectively 0)
        -8.2182481458667263e-13,  # c8
    ]
    RELU_SMOOTH_SCALE = 5.0
    RELU_SMOOTH_MAX_ERROR = 3.997e-15
    
    # Legacy aliases for backward compatibility with PrivateInference.py
    SIGMOID_CHEBYSHEV = SIGMOID
    SWISH_CHEBYSHEV = SWISH
    
    @staticmethod
    def to_native_list(coeffs) -> List[float]:
        """Convert coefficients to native Python list for DESILO API"""
        if isinstance(coeffs, np.ndarray):
            return coeffs.tolist()
        return list(coeffs)
    
    @staticmethod
    def get_coefficients_info() -> Dict[str, Dict]:
        """Get information about all available coefficient sets"""
        return {
            'GELU': {
                'degree': 8,
                'scale': ChebyshevActivations.GELU_SCALE,
                'max_error': ChebyshevActivations.GELU_MAX_ERROR,
                'reference': 'Hendrycks & Gimpel (2016), arXiv:1606.08415',
                'method': 'numpy.polynomial.chebyshev.chebfit'
            },
            'SIGMOID': {
                'degree': 8,
                'scale': ChebyshevActivations.SIGMOID_SCALE,
                'max_error': ChebyshevActivations.SIGMOID_MAX_ERROR,
                'reference': 'Kim et al. (2018), BMC Medical Genomics',
                'method': 'numpy.polynomial.chebyshev.chebfit'
            },
            'SWISH': {
                'degree': 8,
                'scale': ChebyshevActivations.SWISH_SCALE,
                'max_error': ChebyshevActivations.SWISH_MAX_ERROR,
                'reference': 'Ramachandran et al. (2017), arXiv:1710.05941',
                'method': 'numpy.polynomial.chebyshev.chebfit'
            },
            'TANH': {
                'degree': 8,
                'scale': ChebyshevActivations.TANH_SCALE,
                'max_error': ChebyshevActivations.TANH_MAX_ERROR,
                'reference': 'Standard hyperbolic tangent',
                'method': 'numpy.polynomial.chebyshev.chebfit'
            },
            'EXP': {
                'degree': 8,
                'scale': ChebyshevActivations.EXP_SCALE,
                'max_error': ChebyshevActivations.EXP_MAX_ERROR,
                'reference': 'Cheon et al. (2018), ePrint 2018/931',
                'method': 'numpy.polynomial.chebyshev.chebfit'
            },
            'RELU_SMOOTH': {
                'degree': 8,
                'scale': ChebyshevActivations.RELU_SMOOTH_SCALE,
                'max_error': ChebyshevActivations.RELU_SMOOTH_MAX_ERROR,
                'reference': 'Glorot et al. (2011)',
                'method': 'numpy.polynomial.chebyshev.chebfit'
            }
        }


# =============================================================================
# VERSION INFO
# =============================================================================

__version__ = "2.1.0"
__author__ = "Amon"
__doc_version__ = "DESILO FHE v5.5.0+"


if __name__ == "__main__":
    print(f"DESILO FHE Engine Wrapper v{__version__}")
    print(f"Based on DESILO FHE Documentation: https://fhe.desilo.dev/latest/")
    print()
    print("Chebyshev Coefficient Information:")
    print("=" * 60)
    for name, info in ChebyshevActivations.get_coefficients_info().items():
        print(f"\n{name}:")
        print(f"  Degree: {info['degree']}")
        print(f"  Scale: {info['scale']}")
        print(f"  Max Error: {info['max_error']:.6e}")
        print(f"  Method: {info['method']}")
        print(f"  Reference: {info['reference']}")
