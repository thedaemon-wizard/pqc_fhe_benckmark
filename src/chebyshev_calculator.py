"""
Chebyshev Polynomial Coefficient Calculator for FHE Activation Functions
=========================================================================

This script calculates the actual Chebyshev polynomial coefficients for
various activation functions used in homomorphic encryption.

Method: Chebyshev-Gauss Quadrature
----------------------------------
For a function f(x) on [-1, 1], the Chebyshev coefficients c_k are:

    c_k = (2/n) * sum_{j=0}^{n-1} f(x_j) * T_k(x_j)

where:
    - x_j = cos(pi * (j + 0.5) / n) are the Chebyshev nodes
    - T_k(x) = cos(k * arccos(x)) is the k-th Chebyshev polynomial
    - c_0 is divided by 2 for normalization

References:
-----------
1. Cheon et al., "Numerical Method for Comparison on Homomorphically 
   Encrypted Numbers" (IEEE Trans. Info. Forensics Security, 2020)
2. Lee et al., "Minimax Approximation of Sign Function by Composite 
   Polynomial for Homomorphic Comparison" (IEEE Access, 2021)
3. Bossuat et al., "Efficient Bootstrapping for Approximate Homomorphic 
   Encryption with Non-Sparse Keys" (EUROCRYPT 2021)
4. FHERMA Challenge Solutions: https://fherma.io/

Author: Amon
Date: December 27, 2025
"""

import numpy as np
from scipy import special
from typing import List, Tuple, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CHEBYSHEV COEFFICIENT CALCULATION
# =============================================================================

def chebyshev_nodes(n: int) -> np.ndarray:
    """
    Generate Chebyshev nodes (roots of T_n(x))
    
    x_j = cos(pi * (j + 0.5) / n) for j = 0, 1, ..., n-1
    
    Reference: Numerical Recipes, Chapter 5.8
    """
    j = np.arange(n)
    return np.cos(np.pi * (j + 0.5) / n)


def chebyshev_polynomial(k: int, x: np.ndarray) -> np.ndarray:
    """
    Evaluate k-th Chebyshev polynomial T_k(x)
    
    T_k(x) = cos(k * arccos(x))
    
    Reference: Abramowitz & Stegun, Chapter 22
    """
    return np.cos(k * np.arccos(x))


def compute_chebyshev_coefficients(
    func: Callable[[np.ndarray], np.ndarray],
    degree: int,
    num_nodes: int = 1000
) -> np.ndarray:
    """
    Compute Chebyshev coefficients for a function using Chebyshev-Gauss quadrature
    
    Args:
        func: Function to approximate (must accept numpy array)
        degree: Maximum degree of Chebyshev polynomial
        num_nodes: Number of quadrature nodes (higher = more accurate)
    
    Returns:
        Array of Chebyshev coefficients [c_0, c_1, ..., c_degree]
    
    Reference: 
        - Trefethen, "Approximation Theory and Approximation Practice" (2013)
        - Mason & Handscomb, "Chebyshev Polynomials" (2002)
    """
    # Generate Chebyshev nodes
    x = chebyshev_nodes(num_nodes)
    
    # Evaluate function at nodes
    f_values = func(x)
    
    # Compute coefficients using discrete orthogonality
    coeffs = np.zeros(degree + 1)
    
    for k in range(degree + 1):
        T_k = chebyshev_polynomial(k, x)
        # Chebyshev-Gauss quadrature formula
        coeffs[k] = (2.0 / num_nodes) * np.sum(f_values * T_k)
    
    # Normalize c_0
    coeffs[0] /= 2.0
    
    return coeffs


def evaluate_chebyshev(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Evaluate Chebyshev polynomial at points x using Clenshaw recurrence
    
    More numerically stable than direct evaluation.
    
    Reference: Numerical Recipes, Chapter 5.8
    """
    n = len(coeffs)
    if n == 0:
        return np.zeros_like(x)
    if n == 1:
        return np.full_like(x, coeffs[0])
    
    # Clenshaw recurrence
    b_k_plus_2 = np.zeros_like(x)
    b_k_plus_1 = np.zeros_like(x)
    
    for k in range(n - 1, 0, -1):
        b_k = coeffs[k] + 2.0 * x * b_k_plus_1 - b_k_plus_2
        b_k_plus_2 = b_k_plus_1
        b_k_plus_1 = b_k
    
    return coeffs[0] + x * b_k_plus_1 - b_k_plus_2


def compute_approximation_error(
    func: Callable,
    coeffs: np.ndarray,
    num_test_points: int = 10000
) -> Tuple[float, float, float]:
    """
    Compute approximation error statistics
    
    Returns:
        Tuple of (max_error, mean_error, rmse)
    """
    x = np.linspace(-1, 1, num_test_points)
    exact = func(x)
    approx = evaluate_chebyshev(coeffs, x)
    
    errors = np.abs(exact - approx)
    
    max_error = np.max(errors)
    mean_error = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))
    
    return max_error, mean_error, rmse


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU activation function
    
    GELU(x) = x * Phi(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    
    Reference: Hendrycks & Gimpel, "Gaussian Error Linear Units" (2016)
    """
    return x * 0.5 * (1.0 + special.erf(x / np.sqrt(2.0)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function
    
    sigma(x) = 1 / (1 + exp(-x))
    
    Reference: Standard logistic function
    """
    return 1.0 / (1.0 + np.exp(-x))


def swish(x: np.ndarray) -> np.ndarray:
    """
    Swish activation function
    
    swish(x) = x * sigma(x) = x / (1 + exp(-x))
    
    Reference: Ramachandran et al., "Searching for Activation Functions" (2017)
    """
    return x * sigmoid(x)


def tanh_func(x: np.ndarray) -> np.ndarray:
    """
    Hyperbolic tangent
    
    tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    return np.tanh(x)


def softplus(x: np.ndarray) -> np.ndarray:
    """
    Softplus activation function
    
    softplus(x) = log(1 + exp(x))
    
    Reference: Smooth approximation to ReLU
    """
    # Numerically stable version
    return np.where(x > 20, x, np.log1p(np.exp(x)))


def exp_func(x: np.ndarray) -> np.ndarray:
    """
    Exponential function (for softmax)
    
    exp(x) on [-1, 1]
    """
    return np.exp(x)


def relu_smooth(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Smooth ReLU approximation
    
    For Chebyshev approximation, we use softplus as smooth ReLU.
    
    Reference: Lee et al., "Privacy-Preserving Machine Learning with 
    Fully Homomorphic Encryption for Deep Neural Network" (2022)
    """
    # Use softplus scaled to approximate ReLU on [-1, 1]
    return np.log1p(np.exp(x * 5)) / 5  # Scale for [-1, 1] range


def sign_func(x: np.ndarray) -> np.ndarray:
    """
    Sign function (for comparison operations)
    
    sign(x) = -1 if x < 0, 0 if x = 0, 1 if x > 0
    
    Note: For Chebyshev approximation, we use a smooth version
    sign_smooth(x) = x / sqrt(x^2 + epsilon)
    
    Reference: Cheon et al., "Numerical Method for Comparison on 
    Homomorphically Encrypted Numbers" (2020)
    """
    # Smooth sign for better polynomial approximation
    epsilon = 1e-6
    return x / np.sqrt(x**2 + epsilon)


# =============================================================================
# SCALED ACTIVATION FUNCTIONS
# =============================================================================

def create_scaled_function(
    func: Callable,
    input_scale: float
) -> Callable:
    """
    Create a scaled version of activation function for Chebyshev approximation
    
    For input range [-S, S], we compute f(S*x) on [-1, 1]
    """
    def scaled_func(x: np.ndarray) -> np.ndarray:
        return func(input_scale * x)
    return scaled_func


# =============================================================================
# MAIN CALCULATION
# =============================================================================

def calculate_all_coefficients():
    """
    Calculate Chebyshev coefficients for all activation functions
    """
    results = {}
    
    # Configuration for each activation
    activations = {
        'gelu': {
            'func': gelu,
            'input_scales': [1.0, 3.0, 5.0, 8.0],
            'degrees': [4, 6, 8, 12, 16],
            'description': 'GELU activation (Hendrycks & Gimpel, 2016)'
        },
        'sigmoid': {
            'func': sigmoid,
            'input_scales': [1.0, 4.0, 8.0],
            'degrees': [4, 6, 8, 12],
            'description': 'Sigmoid activation'
        },
        'swish': {
            'func': swish,
            'input_scales': [1.0, 3.0, 5.0],
            'degrees': [4, 6, 8, 12],
            'description': 'Swish activation (Ramachandran et al., 2017)'
        },
        'tanh': {
            'func': tanh_func,
            'input_scales': [1.0, 2.0, 4.0],
            'degrees': [4, 6, 8, 12],
            'description': 'Hyperbolic tangent'
        },
        'exp': {
            'func': exp_func,
            'input_scales': [1.0],  # exp(x) on [-1, 1] only
            'degrees': [4, 6, 8, 12],
            'description': 'Exponential function for softmax'
        },
        'softplus': {
            'func': softplus,
            'input_scales': [1.0, 3.0, 5.0],
            'degrees': [4, 6, 8, 12],
            'description': 'Softplus (smooth ReLU)'
        },
        'sign': {
            'func': sign_func,
            'input_scales': [1.0],
            'degrees': [7, 15, 31, 63],  # Odd degrees for odd function
            'description': 'Sign function (Cheon et al., 2020)'
        },
    }
    
    print("=" * 80)
    print("CHEBYSHEV POLYNOMIAL COEFFICIENTS FOR FHE ACTIVATION FUNCTIONS")
    print("=" * 80)
    print()
    print("Method: Chebyshev-Gauss Quadrature with 10000 nodes")
    print("Reference: Trefethen, 'Approximation Theory and Approximation Practice' (2013)")
    print()
    
    for name, config in activations.items():
        print("-" * 80)
        print(f"ACTIVATION: {name.upper()}")
        print(f"Description: {config['description']}")
        print("-" * 80)
        
        results[name] = {}
        
        for input_scale in config['input_scales']:
            print(f"\n  Input scale: {input_scale} (approximates on [-{input_scale}, {input_scale}])")
            
            scaled_func = create_scaled_function(config['func'], input_scale)
            
            for degree in config['degrees']:
                coeffs = compute_chebyshev_coefficients(scaled_func, degree, num_nodes=10000)
                max_err, mean_err, rmse = compute_approximation_error(scaled_func, coeffs)
                
                key = f"scale_{input_scale}_deg_{degree}"
                results[name][key] = {
                    'coefficients': coeffs.tolist(),
                    'input_scale': input_scale,
                    'degree': degree,
                    'max_error': max_err,
                    'mean_error': mean_err,
                    'rmse': rmse
                }
                
                # Print results
                print(f"\n    Degree {degree}:")
                print(f"      Max Error: {max_err:.6e}")
                print(f"      Mean Error: {mean_err:.6e}")
                print(f"      RMSE: {rmse:.6e}")
                print(f"      Coefficients:")
                for i, c in enumerate(coeffs):
                    if abs(c) > 1e-10:  # Only print significant coefficients
                        print(f"        c[{i}] = {c:.15e}")
    
    return results


def generate_python_code(results: dict) -> str:
    """
    Generate Python code with calculated coefficients
    """
    code = '''"""
Chebyshev Polynomial Coefficients for FHE Activation Functions
===============================================================

IMPORTANT: These coefficients were calculated using Chebyshev-Gauss quadrature
with 10000 nodes. They are mathematically derived, not placeholder values.

Calculation Method:
-------------------
For function f(x) on [-1, 1]:
    c_k = (2/n) * sum_{j=0}^{n-1} f(x_j) * T_k(x_j)
    
where x_j = cos(pi * (j + 0.5) / n) are Chebyshev nodes.

For scaled functions on [-S, S], we compute coefficients for f(S*x).
To use: scale input by 1/S, apply polynomial, scale output appropriately.

References:
-----------
1. Trefethen, "Approximation Theory and Approximation Practice" (SIAM, 2013)
2. Cheon et al., "Numerical Method for Comparison on HE Numbers" (2020)
3. Lee et al., "Minimax Approximation of Sign Function" (IEEE Access, 2021)
4. Bossuat et al., "Efficient Bootstrapping for Approximate HE" (EUROCRYPT 2021)

Generated: December 27, 2025
"""

from typing import List, Dict


class ChebyshevActivations:
    """
    Chebyshev polynomial coefficients for FHE activation functions.
    
    All coefficients are computed using Chebyshev-Gauss quadrature.
    Input values should be scaled to [-1, 1] before applying the polynomial.
    
    Usage:
        # For GELU on input range [-5, 5]:
        input_scaled = x / 5.0
        output = evaluate_chebyshev(input_scaled, ChebyshevActivations.GELU_SCALE_5_DEG_8)
        # output is GELU(x)
    """
    
'''
    
    for name, configs in results.items():
        code += f"    # {'='*70}\n"
        code += f"    # {name.upper()} ACTIVATION\n"
        code += f"    # {'='*70}\n\n"
        
        for key, data in configs.items():
            var_name = f"{name.upper()}_{key.upper()}"
            coeffs = data['coefficients']
            scale = data['input_scale']
            degree = data['degree']
            max_err = data['max_error']
            
            code += f"    # Input scale: {scale}, Degree: {degree}, Max Error: {max_err:.2e}\n"
            code += f"    {var_name} = [\n"
            for i, c in enumerate(coeffs):
                code += f"        {c:.15e},  # c[{i}]\n"
            code += f"    ]\n"
            code += f"    {var_name}_SCALE = {scale}\n"
            code += f"    {var_name}_ERROR = {max_err:.6e}\n\n"
    
    code += '''
    # Recommended configurations (balance of precision and computation)
    GELU_RECOMMENDED = GELU_SCALE_5_0_DEG_8
    GELU_RECOMMENDED_SCALE = 5.0
    
    SIGMOID_RECOMMENDED = SIGMOID_SCALE_8_0_DEG_8
    SIGMOID_RECOMMENDED_SCALE = 8.0
    
    SWISH_RECOMMENDED = SWISH_SCALE_5_0_DEG_8
    SWISH_RECOMMENDED_SCALE = 5.0
    
    EXP_RECOMMENDED = EXP_SCALE_1_0_DEG_8
    EXP_RECOMMENDED_SCALE = 1.0


def evaluate_chebyshev_clenshaw(coeffs: List[float], x: float) -> float:
    """
    Evaluate Chebyshev polynomial using Clenshaw recurrence.
    
    Numerically stable evaluation method.
    
    Args:
        coeffs: Chebyshev coefficients [c_0, c_1, ..., c_n]
        x: Input value in [-1, 1]
    
    Returns:
        Sum of c_k * T_k(x)
    
    Reference: Numerical Recipes, Chapter 5.8
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
'''
    
    return code


def main():
    """Main entry point"""
    logger.info("Starting Chebyshev coefficient calculation...")
    
    # Calculate all coefficients
    results = calculate_all_coefficients()
    
    # Generate Python code
    print("\n" + "=" * 80)
    print("GENERATED PYTHON CODE")
    print("=" * 80 + "\n")
    
    code = generate_python_code(results)
    print(code)
    
    # Save to file
    output_file = "chebyshev_coefficients.py"
    with open(output_file, 'w') as f:
        f.write(code)
    
    print(f"\nCoefficients saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()
