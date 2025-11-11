# PINN Exercise Test Suite

This directory contains a PINN (Physics-Informed Neural Network) exercise with an automated test suite to help students validate their implementations.

## Files

- **`First_MP_exercise_with_code.ipynb`**: Student exercise notebook with scaffolding and TODOs
- **`First_MP_exercise_solved.ipynb`**: Complete solution notebook
- **`test_pinn.py`**: Automated test suite for validation
- **`TEST_SUITE_README.md`**: This file

## Test Suite Overview

The test suite (`test_pinn.py`) provides comprehensive validation of student implementations including:

### Test Categories

1. **Geometry Class Tests** (`test_geometry`)
   - Proper initialization of domain parameters
   - Correct sampling of interior points
   - Correct sampling of boundary points  
   - Correct sampling of initial condition points
   - Optional: sampling at specific times
   - Verification of output shapes and value ranges
   - Checking randomness of sampling

2. **PINN Class Tests** (`test_pinn`)
   - Proper network initialization
   - Forward pass functionality
   - Correct input/output shapes
   - Parameter initialization quality (Glorot/Xavier)
   - No NaN or Inf values in outputs

3. **Loss Function Tests** (`test_loss_functions`)
   - PDE residual loss (`loss_pde`)
   - Boundary condition loss (`loss_bc`)
   - Initial condition loss (`loss_ic`)
   - Total loss computation and consistency
   - Verification of scalar outputs
   - Checking for NaN/Inf values

4. **Training Function Tests** (`test_train_function`) - Optional
   - Basic training loop functionality
   - Correct return values
   - Loss decrease verification

### Running Tests

#### In the Student Notebook

Tests are integrated throughout the notebook. After implementing each component, students run:

```python
# Test Geometry class
from test_pinn import test_geometry
test_geometry(Geometry, verbose=True)

# Test PINN class  
from test_pinn import test_pinn
test_pinn(Pinn, verbose=True)

# Test loss functions
from test_pinn import test_loss_functions
test_loss_functions(loss_pde, loss_bc, loss_ic, total_loss, Pinn, Geometry, verbose=True)

# Run all tests
from test_pinn import run_all_tests
all_passed = run_all_tests(Geometry, Pinn, loss_pde, loss_bc, loss_ic, total_loss, verbose=True)
```

#### Standalone Testing

You can also run tests outside the notebook:

```python
from test_pinn import *
from First_MP_exercise_solved import Geometry, Pinn, loss_pde, loss_bc, loss_ic, total_loss

# Run comprehensive tests
run_all_tests(Geometry, Pinn, loss_pde, loss_bc, loss_ic, total_loss, verbose=True)
```

## Test Output Format

Tests provide colored, structured output:

```
======================================================================
TEST SUMMARY
======================================================================

✅ PASSED (X tests):
   • Test name 1
   • Test name 2
   ...

⚠️  WARNINGS (Y):
   • Test name: Warning message
   ...

❌ FAILED (Z tests):
   • Test name
     Error: Detailed error message
   ...

Total: X/Y tests passed
======================================================================
```

## Implementation Guide for Students

### Step 1: Implement Geometry Class

Complete the methods:
- `__init__()`: Initialize domain parameters
- `sample_interior()`: Sample random points in interior domain
- `sample_boundary()`: Sample points on spatial boundaries
- `sample_initial()`: Sample points at initial time
- `sample_at_time()`: (Optional) Sample points at specific time

**Test after completion:**
```python
from test_pinn import test_geometry
test_geometry(Geometry, verbose=True)
```

### Step 2: Implement PINN Class

Complete the methods:
- `initialize_params()`: Initialize network weights with Glorot uniform
- `neural_net()`: Forward pass through network with tanh activations
- `__call__()`: Make network callable

**Test after completion:**
```python
from test_pinn import test_pinn
test_pinn(Pinn, verbose=True)
```

### Step 3: Implement Loss Functions

Complete the functions:
- `loss_pde()`: PDE residual loss using automatic differentiation
- `loss_bc()`: Boundary condition loss
- `loss_ic()`: Initial condition loss
- `total_loss()`: Combined loss with individual components

**Test after completion:**
```python
from test_pinn import test_loss_functions
test_loss_functions(loss_pde, loss_bc, loss_ic, total_loss, Pinn, Geometry, verbose=True)
```

### Step 4: Comprehensive Validation

Before implementing training:
```python
from test_pinn import run_all_tests
all_passed = run_all_tests(Geometry, Pinn, loss_pde, loss_bc, loss_ic, total_loss, verbose=True)
```

All tests should pass before proceeding to training!

### Step 5: Implement Training Loop

Complete `train_pinn()`:
- Initialize optimizer
- Sample points each epoch
- Compute loss and gradients
- Update parameters
- Track and return loss history

## Common Issues and Solutions

### Issue: "Method not implemented"
**Solution**: Complete the TODO sections in the code. Remove `raise NotImplementedError()` statements.

### Issue: Shape mismatch errors
**Solution**: Check that:
- Geometry methods return `(N, 2)` arrays
- PINN outputs `(N, 1)` arrays
- Loss functions return scalars

### Issue: NaN or Inf values
**Solution**: Check:
- Network initialization (use Glorot uniform)
- Learning rate (try smaller values like 1e-4)
- Gradient computation in loss_pde

### Issue: Loss not decreasing
**Solution**:
- Verify all loss components are computed correctly
- Check that automatic differentiation is working
- Try different network architectures or learning rates

## Test Suite Customization

Instructors can modify `test_pinn.py` to:
- Add additional tests
- Adjust tolerance levels
- Change test point sampling sizes
- Add domain-specific validation

## Dependencies

The test suite requires:
- JAX and jax.numpy
- NumPy  
- The student's implementations (Geometry, Pinn, loss functions)

## Feedback and Improvements

The test suite provides:
- ✅ Clear pass/fail indicators
- 📊 Quantitative metrics where applicable
- 💡 Helpful error messages
- ⚠️ Warnings for optional features or potential issues

## Credits

Test suite designed for the Physics-Informed Neural Networks exercise at [Your Institution].
