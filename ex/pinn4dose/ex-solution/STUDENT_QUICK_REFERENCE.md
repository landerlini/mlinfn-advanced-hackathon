# PINN Exercise - Quick Reference Guide

## 🎯 Getting Started

1. Open `First_MP_exercise_with_code.ipynb`
2. Read the introduction and task descriptions
3. Implement code where you see `TODO` comments
4. Run test cells to validate your work
5. Fix errors and re-test until all tests pass

## 📋 Implementation Checklist

### ✅ Task 1: Geometry Class
- [ ] Implement `sample_interior(key, N1)`
- [ ] Implement `sample_boundary(key, N2)`
- [ ] Implement `sample_initial(key, N3)`
- [ ] (Optional) Implement `sample_at_time(key, N4, t1)`
- [ ] Run geometry tests - all should pass

### ✅ Task 2: PINN Class
- [ ] Implement `initialize_params(key, layers)`
- [ ] Implement `neural_net(params, x)`
- [ ] Run PINN tests - all should pass

### ✅ Task 3: Loss Functions
- [ ] Implement `loss_pde(params, pinn, points, D)`
- [ ] Implement `loss_bc(params, pinn, points)`
- [ ] Implement `loss_ic(params, pinn, points, L)`
- [ ] Implement `total_loss(...)`
- [ ] Run loss function tests - all should pass

### ✅ Task 4: Comprehensive Validation
- [ ] Run `run_all_tests()` - everything should pass
- [ ] If not, debug and fix issues

### ✅ Task 5: Training
- [ ] Implement `train_pinn(...)`
- [ ] Run training for 1000-5000 epochs
- [ ] Observe loss decreasing
- [ ] Visualize training curves

### ✅ Task 6: Evaluation
- [ ] Compare predictions with analytical solution
- [ ] Calculate MAE, max error
- [ ] Visualize heatmaps and time slices
- [ ] Analyze results

## 🔑 Key Functions to Know

### JAX Random
```python
# Create random key
key = random.PRNGKey(seed)

# Split key for multiple random operations
key, subkey1, subkey2 = random.split(key, 3)

# Sample uniform random values
x = random.uniform(key, shape=(N, 1), minval=0.0, maxval=L)
```

### JAX Arrays
```python
# Create arrays
jnp.zeros((N, 1))           # Array of zeros
jnp.full((N, 1), value)     # Array filled with value
jnp.linspace(start, end, N) # Evenly spaced values

# Stack arrays
jnp.hstack((a, b))          # Horizontal stack
jnp.vstack((a, b))          # Vertical stack
```

### JAX Automatic Differentiation
```python
from jax import grad

# First derivative with respect to argument 0 (x)
df_dx = grad(f, argnums=0)(x, t)

# First derivative with respect to argument 1 (t)
df_dt = grad(f, argnums=1)(x, t)

# Second derivative
d2f_dx2 = grad(grad(f, argnums=0), argnums=0)(x, t)
```

### Neural Network Operations
```python
# Linear transformation
output = jnp.dot(input, W) + b

# Activation function
output = jnp.tanh(input)

# Forward pass pattern
for W, b in hidden_layers:
    x = jnp.tanh(jnp.dot(x, W) + b)
# Last layer (no activation)
W, b = output_layer
x = jnp.dot(x, W) + b
```

## 🧪 Running Tests

### Individual Component Tests
```python
# Test Geometry
from test_pinn import test_geometry
test_geometry(Geometry, verbose=True)

# Test PINN
from test_pinn import test_pinn
test_pinn(Pinn, verbose=True)

# Test Loss Functions
from test_pinn import test_loss_functions
test_loss_functions(loss_pde, loss_bc, loss_ic, total_loss, 
                   Pinn, Geometry, verbose=True)
```

### Comprehensive Test
```python
from test_pinn import run_all_tests
all_passed = run_all_tests(Geometry, Pinn, loss_pde, loss_bc, 
                           loss_ic, total_loss, verbose=True)
```

## 🐛 Common Errors and Solutions

### Error: Shape mismatch
```
Expected shape (N, 2), got (N,)
```
**Solution**: Use `x[:, None]` or `x.reshape(-1, 1)` to add dimension

### Error: Can't split random key
```
TypeError: split() takes 2 positional arguments but 3 were given
```
**Solution**: 
```python
# Wrong
key1, key2 = random.split(key, 2)

# Correct
key, key1, key2 = random.split(key, 3)
# Or
key1, key2 = random.split(key)
```

### Error: Gradient returns None
```
TypeError: Gradient of None
```
**Solution**: Make sure the function returns a scalar value (not an array)

### Error: NotImplementedError
```
NotImplementedError: Implement sample_interior method
```
**Solution**: You haven't implemented this method yet - complete the TODO

## 📊 Expected Results

After training (1000 epochs, may vary):
- **Total Loss**: < 0.01 (lower is better)
- **MAE**: < 0.01 (compared to analytical solution)
- **Training Time**: 1-5 minutes (depends on hardware)

## 💡 Tips for Success

1. **Start Simple**: Implement one method at a time, test immediately
2. **Read Docstrings**: Function signatures tell you what to return
3. **Check Shapes**: Use `print(array.shape)` liberally during development
4. **Test Small**: Use small N values (e.g., N=10) while debugging
5. **Visualize**: Plot sampled points to verify correctness
6. **Read Errors**: Error messages usually point to the exact problem
7. **Use Hints**: TODO comments contain implementation hints

## 📚 Useful Resources

### JAX Documentation
- [JAX Quickstart](https://jax.readthedocs.io/en/latest/quickstart.html)
- [JAX Random Module](https://jax.readthedocs.io/en/latest/jax.random.html)
- [JAX Automatic Differentiation](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)

### PINN Resources
- Original paper: [Physics-informed neural networks](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- Tutorial: [DeepXDE Documentation](https://deepxde.readthedocs.io/)

## 🎓 Learning Objectives

By completing this exercise, you will:
- ✅ Understand PINN architecture and components
- ✅ Implement domain sampling for PDE problems
- ✅ Build neural networks in JAX
- ✅ Use automatic differentiation for PDE residuals
- ✅ Construct physics-informed loss functions
- ✅ Train neural networks with custom losses
- ✅ Evaluate and visualize PDE solutions
- ✅ Compare numerical methods (PINNs vs FTCS)

## ❓ Getting Help

If you're stuck:
1. **Check test output**: Error messages are usually informative
2. **Review docstrings**: They explain expected inputs/outputs
3. **Read TODO hints**: Implementation guidance in comments
4. **Compare shapes**: Use `print()` to debug array dimensions
5. **Consult solution**: `First_MP_exercise_solved.ipynb` (last resort!)

## 🎉 Completion

Once all tests pass and your PINN trains successfully:
- Compare your results with the analytical solution
- Experiment with different architectures
- Try the optional FTCS comparison
- Attempt the inverse problem challenge

Good luck! 🚀
