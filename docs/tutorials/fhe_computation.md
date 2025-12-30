# FHE Computation Tutorial

This tutorial demonstrates Fully Homomorphic Encryption (FHE) using the CKKS scheme for privacy-preserving computation on encrypted data.

## Overview

CKKS (Cheon-Kim-Kim-Song) enables approximate arithmetic on encrypted floating-point numbers. This tutorial covers:

1. Understanding CKKS parameters
2. Encrypting numerical data
3. Performing homomorphic operations
4. Decrypting results
5. Managing computational depth
6. Real-world applications

## Prerequisites

```bash
# Install the library
pip install pqc-fhe-lib

# Verify installation
python -c "from pqc_fhe_integration import FHEEngine; print('OK')"
```

## CKKS Parameters Explained

| Parameter | Description | Impact |
|-----------|-------------|--------|
| `poly_modulus_degree` | Ring dimension (N) | Security & capacity |
| `coeff_mod_bit_sizes` | Coefficient modulus chain | Multiplication depth |
| `scale` | Encoding precision | Accuracy vs. noise budget |

**Default Configuration**:
- `poly_modulus_degree`: 8192 (good balance)
- `coeff_mod_bit_sizes`: [60, 40, 40, 60] (3 multiplications)
- `scale`: 2^40 (sufficient for most applications)

## Step 1: Basic Encryption and Decryption

```python
from pqc_fhe_integration import FHEEngine

# Initialize FHE engine
fhe = FHEEngine()

# Encrypt a single value
plaintext_value = 3.14159
ciphertext = fhe.encrypt(plaintext_value)

print(f"Original: {plaintext_value}")
print(f"Ciphertext size: ~{len(str(ciphertext))} characters")

# Decrypt
decrypted_value = fhe.decrypt(ciphertext)
print(f"Decrypted: {decrypted_value:.5f}")
print(f"Error: {abs(plaintext_value - decrypted_value):.2e}")
```

### Encrypting Arrays

```python
import numpy as np

# Encrypt an array of values
data = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
ciphertext = fhe.encrypt(data.tolist())

# Decrypt
decrypted = fhe.decrypt(ciphertext)
print(f"Original:  {data}")
print(f"Decrypted: {np.array(decrypted[:len(data)])}")
```

## Step 2: Homomorphic Operations

### Addition

```python
# Encrypt two values
ct_a = fhe.encrypt(10.0)
ct_b = fhe.encrypt(5.0)

# Add encrypted values (no decryption needed!)
ct_sum = fhe.add(ct_a, ct_b)

# Decrypt result
result = fhe.decrypt(ct_sum)
print(f"10.0 + 5.0 = {result:.2f}")  # Expected: 15.0
```

### Subtraction

```python
ct_diff = fhe.subtract(ct_a, ct_b)
result = fhe.decrypt(ct_diff)
print(f"10.0 - 5.0 = {result:.2f}")  # Expected: 5.0
```

### Multiplication

```python
ct_product = fhe.multiply(ct_a, ct_b)
result = fhe.decrypt(ct_product)
print(f"10.0 * 5.0 = {result:.2f}")  # Expected: 50.0
```

### Scalar Operations

```python
# Multiply by plaintext scalar (more efficient)
ct_scaled = fhe.multiply_scalar(ct_a, 3.0)
result = fhe.decrypt(ct_scaled)
print(f"10.0 * 3 = {result:.2f}")  # Expected: 30.0

# Add plaintext scalar
ct_shifted = fhe.add_scalar(ct_a, 100.0)
result = fhe.decrypt(ct_shifted)
print(f"10.0 + 100 = {result:.2f}")  # Expected: 110.0
```

### Negation

```python
ct_neg = fhe.negate(ct_a)
result = fhe.decrypt(ct_neg)
print(f"-10.0 = {result:.2f}")  # Expected: -10.0
```

### Square

```python
ct_square = fhe.square(ct_a)
result = fhe.decrypt(ct_square)
print(f"10.0^2 = {result:.2f}")  # Expected: 100.0
```

## Step 3: Complex Computations

### Polynomial Evaluation

Compute f(x) = ax^2 + bx + c on encrypted x:

```python
def encrypted_polynomial(fhe: FHEEngine, ct_x, a: float, b: float, c: float):
    """
    Evaluate polynomial f(x) = ax^2 + bx + c on encrypted x.
    
    Operations:
    1. x^2 (square)
    2. ax^2 (scalar multiply)
    3. bx (scalar multiply)
    4. ax^2 + bx (add)
    5. ax^2 + bx + c (add scalar)
    """
    # x^2
    ct_x_squared = fhe.square(ct_x)
    
    # ax^2
    ct_ax2 = fhe.multiply_scalar(ct_x_squared, a)
    
    # bx
    ct_bx = fhe.multiply_scalar(ct_x, b)
    
    # ax^2 + bx
    ct_sum = fhe.add(ct_ax2, ct_bx)
    
    # ax^2 + bx + c
    ct_result = fhe.add_scalar(ct_sum, c)
    
    return ct_result


# Example: f(x) = 2x^2 + 3x + 1
x = 5.0
ct_x = fhe.encrypt(x)

ct_result = encrypted_polynomial(fhe, ct_x, a=2, b=3, c=1)
result = fhe.decrypt(ct_result)

expected = 2 * x**2 + 3 * x + 1
print(f"f({x}) = 2*{x}^2 + 3*{x} + 1")
print(f"Expected: {expected}")
print(f"Computed: {result:.2f}")
print(f"Error: {abs(expected - result):.2e}")
```

### Mean Calculation

```python
def encrypted_mean(fhe: FHEEngine, ciphertexts: list):
    """
    Compute mean of encrypted values.
    
    Operations: n-1 additions + 1 scalar multiplication
    """
    n = len(ciphertexts)
    
    # Sum all ciphertexts
    ct_sum = ciphertexts[0]
    for ct in ciphertexts[1:]:
        ct_sum = fhe.add(ct_sum, ct)
    
    # Divide by n (multiply by 1/n)
    ct_mean = fhe.multiply_scalar(ct_sum, 1.0 / n)
    
    return ct_mean


# Example
values = [10.0, 20.0, 30.0, 40.0, 50.0]
ciphertexts = [fhe.encrypt(v) for v in values]

ct_mean = encrypted_mean(fhe, ciphertexts)
result = fhe.decrypt(ct_mean)

print(f"Values: {values}")
print(f"Expected mean: {sum(values)/len(values)}")
print(f"Encrypted mean: {result:.2f}")
```

### Variance Calculation

```python
def encrypted_variance(fhe: FHEEngine, ciphertexts: list, known_mean: float):
    """
    Compute variance: Var = E[X^2] - E[X]^2
    
    Since we need mean squared, we use known mean.
    For fully encrypted variance, additional techniques needed.
    """
    n = len(ciphertexts)
    
    # Compute sum of (x - mean)^2
    # We'll compute sum of x^2 and adjust
    
    # Sum of x^2
    ct_sum_sq = fhe.square(ciphertexts[0])
    for ct in ciphertexts[1:]:
        ct_sq = fhe.square(ct)
        ct_sum_sq = fhe.add(ct_sum_sq, ct_sq)
    
    # E[X^2]
    ct_e_x2 = fhe.multiply_scalar(ct_sum_sq, 1.0 / n)
    
    # Subtract mean^2 (plaintext since mean is known)
    mean_squared = known_mean ** 2
    ct_variance = fhe.add_scalar(ct_e_x2, -mean_squared)
    
    return ct_variance


# Example
values = [10.0, 20.0, 30.0, 40.0, 50.0]
mean = sum(values) / len(values)
ciphertexts = [fhe.encrypt(v) for v in values]

ct_var = encrypted_variance(fhe, ciphertexts, mean)
result = fhe.decrypt(ct_var)

expected_var = sum((x - mean)**2 for x in values) / len(values)
print(f"Expected variance: {expected_var}")
print(f"Encrypted variance: {result:.2f}")
```

## Step 4: Managing Computational Depth

CKKS has a limited multiplication depth. Each multiplication consumes noise budget.

### Checking Noise Budget

```python
def demonstrate_depth_consumption():
    """Show how multiplication depth is consumed."""
    
    fhe = FHEEngine()
    ct = fhe.encrypt(2.0)
    
    print("Multiplication chain: 2 -> 4 -> 16 -> 256")
    print("-" * 40)
    
    results = [2.0]
    
    for i in range(3):
        ct = fhe.square(ct)
        decrypted = fhe.decrypt(ct)
        expected = results[-1] ** 2
        results.append(expected)
        
        error = abs(expected - decrypted)
        print(f"Step {i+1}: Expected {expected:.0f}, Got {decrypted:.2f}, Error: {error:.2e}")
    
    # Note: After 3-4 multiplications, errors may increase significantly


demonstrate_depth_consumption()
```

### Relinearization and Rescaling

```python
"""
CKKS multiplication increases ciphertext size.
Relinearization reduces size back to normal.
Rescaling maintains precision after multiplication.

In our FHEEngine, these are handled automatically.
"""

# Example showing automatic handling
ct_a = fhe.encrypt(100.0)
ct_b = fhe.encrypt(0.5)

# Multiple multiplications
ct_result = fhe.multiply(ct_a, ct_b)  # 50.0
ct_result = fhe.multiply(ct_result, ct_b)  # 25.0
ct_result = fhe.multiply(ct_result, ct_b)  # 12.5

result = fhe.decrypt(ct_result)
print(f"100 * 0.5^3 = {result:.2f}")  # Expected: 12.5
```

## Step 5: Batching (SIMD Operations)

CKKS supports Single Instruction Multiple Data (SIMD) operations:

```python
"""
SIMD allows operating on multiple values simultaneously.
This is much more efficient than individual encryptions.
"""

# Encrypt batch of values
batch1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
batch2 = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

ct_batch1 = fhe.encrypt(batch1)
ct_batch2 = fhe.encrypt(batch2)

# Element-wise operations
ct_sum = fhe.add(ct_batch1, ct_batch2)
ct_product = fhe.multiply(ct_batch1, ct_batch2)

# Decrypt
sum_result = fhe.decrypt(ct_sum)[:len(batch1)]
product_result = fhe.decrypt(ct_product)[:len(batch1)]

print("Batch 1:", batch1)
print("Batch 2:", batch2)
print("Sum:", [f"{x:.1f}" for x in sum_result])
print("Product:", [f"{x:.1f}" for x in product_result])

# Verify
expected_sum = [a + b for a, b in zip(batch1, batch2)]
expected_product = [a * b for a, b in zip(batch1, batch2)]
print("\nExpected sum:", expected_sum)
print("Expected product:", expected_product)
```

## Step 6: Real-World Application - Private Statistics

### Private Salary Statistics

```python
"""
Compute statistics on encrypted salary data.
Company never sees individual salaries!
"""

class PrivateSalaryAnalyzer:
    """Analyze salary data without exposing individual values."""
    
    def __init__(self):
        self.fhe = FHEEngine()
        self.encrypted_salaries = []
    
    def add_salary(self, salary: float):
        """Employee submits encrypted salary."""
        ct = self.fhe.encrypt(salary)
        self.encrypted_salaries.append(ct)
    
    def compute_total(self):
        """Compute total payroll (encrypted)."""
        if not self.encrypted_salaries:
            return None
        
        ct_total = self.encrypted_salaries[0]
        for ct in self.encrypted_salaries[1:]:
            ct_total = self.fhe.add(ct_total, ct)
        
        return ct_total
    
    def compute_average(self):
        """Compute average salary (encrypted)."""
        ct_total = self.compute_total()
        if ct_total is None:
            return None
        
        n = len(self.encrypted_salaries)
        ct_avg = self.fhe.multiply_scalar(ct_total, 1.0 / n)
        return ct_avg
    
    def decrypt_result(self, ciphertext) -> float:
        """Decrypt aggregate result (only used for final output)."""
        return self.fhe.decrypt(ciphertext)


# Demonstration
def demonstrate_private_salary():
    analyzer = PrivateSalaryAnalyzer()
    
    # Employees submit encrypted salaries
    salaries = [75000, 85000, 95000, 65000, 105000]
    print("Employee salaries (encrypted individually):")
    for i, salary in enumerate(salaries):
        print(f"  Employee {i+1}: ${salary:,} (encrypted)")
        analyzer.add_salary(float(salary))
    
    # Company computes statistics without seeing individual values
    print("\nComputing on encrypted data...")
    
    ct_total = analyzer.compute_total()
    ct_average = analyzer.compute_average()
    
    # Only aggregate results are decrypted
    total = analyzer.decrypt_result(ct_total)
    average = analyzer.decrypt_result(ct_average)
    
    print(f"\nResults:")
    print(f"  Total payroll: ${total:,.2f}")
    print(f"  Average salary: ${average:,.2f}")
    print(f"\nExpected:")
    print(f"  Total: ${sum(salaries):,}")
    print(f"  Average: ${sum(salaries)/len(salaries):,.2f}")


demonstrate_private_salary()
```

### Private Machine Learning Inference

```python
"""
Simple linear model inference on encrypted data.
Model weights are public, input data is private.
"""

class PrivateLinearModel:
    """Linear model with encrypted inference."""
    
    def __init__(self, weights: list, bias: float):
        self.fhe = FHEEngine()
        self.weights = weights
        self.bias = bias
    
    def predict_encrypted(self, encrypted_features: list):
        """
        Compute y = w1*x1 + w2*x2 + ... + wn*xn + b
        on encrypted features.
        """
        if len(encrypted_features) != len(self.weights):
            raise ValueError("Feature count must match weight count")
        
        # Compute weighted sum
        ct_result = self.fhe.multiply_scalar(
            encrypted_features[0], 
            self.weights[0]
        )
        
        for ct_feature, weight in zip(encrypted_features[1:], self.weights[1:]):
            ct_weighted = self.fhe.multiply_scalar(ct_feature, weight)
            ct_result = self.fhe.add(ct_result, ct_weighted)
        
        # Add bias
        ct_result = self.fhe.add_scalar(ct_result, self.bias)
        
        return ct_result
    
    def encrypt_features(self, features: list):
        """Encrypt input features."""
        return [self.fhe.encrypt(f) for f in features]
    
    def decrypt_prediction(self, ct_prediction) -> float:
        """Decrypt prediction result."""
        return self.fhe.decrypt(ct_prediction)


# Demonstration: House price prediction
def demonstrate_private_ml():
    # Model: price = 100*sqft + 50000*bedrooms + 30000*bathrooms + 50000
    weights = [100.0, 50000.0, 30000.0]  # sqft, bedrooms, bathrooms
    bias = 50000.0
    
    model = PrivateLinearModel(weights, bias)
    
    # Private input: 1500 sqft, 3 bedrooms, 2 bathrooms
    features = [1500.0, 3.0, 2.0]
    
    print("House Price Prediction (Private)")
    print("=" * 40)
    print(f"Features (encrypted): {features}")
    print(f"Model weights: {weights}")
    print(f"Bias: {bias}")
    
    # Encrypt features
    encrypted_features = model.encrypt_features(features)
    print(f"\nFeatures encrypted: {len(encrypted_features)} ciphertexts")
    
    # Predict on encrypted data
    ct_prediction = model.predict_encrypted(encrypted_features)
    print("Prediction computed on encrypted data")
    
    # Decrypt result
    prediction = model.decrypt_prediction(ct_prediction)
    
    # Verify
    expected = sum(w * f for w, f in zip(weights, features)) + bias
    
    print(f"\nPredicted price: ${prediction:,.2f}")
    print(f"Expected price: ${expected:,.2f}")
    print(f"Error: ${abs(prediction - expected):,.2f}")


demonstrate_private_ml()
```

## CLI Examples

### Encrypt Data

```bash
# Encrypt values from command line
pqc-fhe fhe-encrypt --values 3.14 2.71 1.41 --output encrypted.bin

# Encrypt from file
echo "100.5,200.3,300.1" > data.csv
pqc-fhe fhe-encrypt --file data.csv --output encrypted.bin

# With JSON output
pqc-fhe fhe-encrypt --values 42.0 --json
```

### Perform Computations

```bash
# Add ciphertexts
pqc-fhe fhe-compute --operation add \
    --input encrypted1.bin \
    --operand encrypted2.bin \
    --output result.bin

# Multiply by scalar
pqc-fhe fhe-compute --operation scalar_multiply \
    --input encrypted.bin \
    --scalar 2.5 \
    --output scaled.bin

# Square
pqc-fhe fhe-compute --operation square \
    --input encrypted.bin \
    --output squared.bin
```

### Decrypt Results

```bash
# Decrypt to stdout
pqc-fhe fhe-decrypt --ciphertext result.bin

# Decrypt to file
pqc-fhe fhe-decrypt --ciphertext result.bin --output plaintext.txt

# JSON output
pqc-fhe fhe-decrypt --ciphertext result.bin --json
```

## Performance Considerations

### Operation Costs

| Operation | Relative Cost | Noise Growth |
|-----------|--------------|--------------|
| Addition | 1x | Low |
| Subtraction | 1x | Low |
| Scalar multiply | 2x | Low |
| Multiplication | 10x | High |
| Square | 8x | High |
| Rotation | 5x | Medium |

### Optimization Tips

1. **Batch operations**: Use SIMD to process multiple values
2. **Minimize multiplications**: Reorganize computations
3. **Use scalar operations**: When one operand is plaintext
4. **Precompute constants**: Encode constants once
5. **Choose appropriate parameters**: Balance security and performance

```python
# Example: Optimized polynomial evaluation
def optimized_polynomial(fhe, ct_x, coefficients):
    """
    Horner's method: fewer multiplications
    f(x) = a + x*(b + x*(c + x*d))
    
    Reduces n multiplications to n-1
    """
    # Reverse coefficients for Horner's method
    result = fhe.encrypt(coefficients[-1])
    
    for coef in reversed(coefficients[:-1]):
        result = fhe.multiply(result, ct_x)
        result = fhe.add_scalar(result, coef)
    
    return result
```

## Error Handling

```python
from pqc_fhe_integration import FHEEngine

fhe = FHEEngine()

try:
    # Encrypt
    ct = fhe.encrypt(3.14)
    
    # Multiple operations that might exceed depth
    for _ in range(10):
        ct = fhe.square(ct)
    
    # Decrypt - might fail if noise budget exhausted
    result = fhe.decrypt(ct)
    
except ValueError as e:
    print(f"Encryption/Decryption error: {e}")
except RuntimeError as e:
    print(f"Computation error (likely noise budget exhausted): {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Next Steps

- [PQC Key Exchange Tutorial](pqc_key_exchange.md) - Secure key establishment
- [Hybrid Workflow Tutorial](hybrid_workflow.md) - Combine PQC with FHE
- [Enterprise Integration Tutorial](enterprise_integration.md) - Production deployment
