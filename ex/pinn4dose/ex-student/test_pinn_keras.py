"""
Test suite for PINN exercise validation.

This module provides automated tests to validate student implementations
of the Physics-Informed Neural Network exercise components.
"""
import numpy as np
import tensorflow as tf


class TestResults:
    """Container for test results with colored output."""
    
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
    
    def add_pass(self, test_name):
        self.passed.append(test_name)
    
    def add_fail(self, test_name, error_msg):
        self.failed.append((test_name, error_msg))
    
    def add_warning(self, test_name, warning_msg):
        self.warnings.append((test_name, warning_msg))
    
    def print_summary(self):
        """Print a summary of all test results."""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        if self.passed:
            print(f"\n✅ PASSED ({len(self.passed)} tests):")
            for test in self.passed:
                print(f"   • {test}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for test, msg in self.warnings:
                print(f"   • {test}: {msg}")
        
        if self.failed:
            print(f"\n❌ FAILED ({len(self.failed)} tests):")
            for test, msg in self.failed:
                print(f"   • {test}")
                print(f"     Error: {msg}")
        
        total = len(self.passed) + len(self.failed)
        print(f"\nTotal: {len(self.passed)}/{total} tests passed")
        print("="*70 + "\n")
        
        return len(self.failed) == 0


# ============================================================================
# Test 1: Geometry Class
# ============================================================================

def test_geometry(Geometry, verbose=True):
    """
    Test the Geometry class implementation.
    
    Parameters:
    -----------
    Geometry : class
        The student's Geometry class
    verbose : bool
        Print detailed test results
    
    Returns:
    --------
    results : TestResults
        Test results object
    """
    results = TestResults()
    
    if verbose:
        print("="*70)
        print("Testing Geometry Class")
        print("="*70 + "\n")
    
    # Test 1.1: Initialization
    try:
        L, t_start, t_end = 10.0, 0.0, 1.0
        geo = Geometry(L=L, t_start=t_start, t_end=t_end)
        
        assert hasattr(geo, 'L'), "Geometry must have attribute 'L'"
        assert hasattr(geo, 't_start'), "Geometry must have attribute 't_start'"
        assert hasattr(geo, 't_end'), "Geometry must have attribute 't_end'"
        assert geo.L == L, f"Expected L={L}, got {geo.L}"
        
        results.add_pass("Geometry initialization")
    except Exception as e:
        results.add_fail("Geometry initialization", str(e))
        return results  # Can't continue without initialization
    
    # Test 1.2: sample_interior
    try:
        N1 = 100
        points = geo.sample_interior(N1)
        
        assert points.shape == (N1, 2), f"Expected shape ({N1}, 2), got {points.shape}"
        
        # Check x is in (0, L)
        x_coords = points[:, 0]
        assert np.all(x_coords >= 0) and np.all(x_coords <= L), \
            "x coordinates should be in [0, L]"
        
        # Check t is in (t_start, t_end)
        t_coords = points[:, 1]
        assert np.all(t_coords >= t_start) and np.all(t_coords <= t_end), \
            "t coordinates should be in [t_start, t_end]"
        
        # Check randomness: different calls should give different points
        points2 = geo.sample_interior( N1)
        assert not np.allclose(points, points2), \
            "Different random keys should produce different samples"
        
        results.add_pass("Geometry.sample_interior()")
    except Exception as e:
        results.add_fail("Geometry.sample_interior()", str(e))
    
    # Test 1.3: sample_boundary
    try:
        N2 = 100
        points = geo.sample_boundary(N2)
        
        assert points.shape == (N2, 2), f"Expected shape ({N2}, 2), got {points.shape}"
        
        # Check x is at boundaries (0 or L)
        x_coords = points[:, 0]
        at_boundaries = np.logical_or(
            np.isclose(x_coords, 0.0, atol=1e-6),
            np.isclose(x_coords, L, atol=1e-6)
        )
        assert np.all(at_boundaries), \
            "All x coordinates should be at boundaries (0 or L)"
        
        # Check t is in (t_start, t_end)
        t_coords = points[:, 1]
        assert np.all(t_coords >= t_start) and np.all(t_coords <= t_end), \
            "t coordinates should be in [t_start, t_end]"
        
        results.add_pass("Geometry.sample_boundary()")
    except Exception as e:
        results.add_fail("Geometry.sample_boundary()", str(e))
    
    # Test 1.4: sample_initial
    try:
        N3 = 100
        points = geo.sample_initial(N3)
        
        assert points.shape == (N3, 2), f"Expected shape ({N3}, 2), got {points.shape}"
        
        # Check x is in [0, L]
        x_coords = points[:, 0]
        assert np.all(x_coords >= 0) and np.all(x_coords <= L), \
            "x coordinates should be in [0, L]"
        
        # Check t is 0
        t_coords = points[:, 1]
        assert np.allclose(t_coords, 0.0, atol=1e-6), \
            "All t coordinates should be 0"
        
        results.add_pass("Geometry.sample_initial()")
    except Exception as e:
        results.add_fail("Geometry.sample_initial()", str(e))
    
    # Test 1.5: sample_at_time (optional)
    if hasattr(geo, 'sample_at_time'):
        try:
            N4 = 50
            t_test = 0.5
            points = geo.sample_at_time(N4, t_test)

            assert points.shape == (N4, 2), f"Expected shape ({N4}, 2), got {points.shape}"
            
            # Check t is t_test
            t_coords = points[:, 1]
            assert np.allclose(t_coords, t_test, atol=1e-6), \
                f"All t coordinates should be {t_test}"
            
            results.add_pass("Geometry.sample_at_time() [OPTIONAL]")
        except Exception as e:
            results.add_warning("Geometry.sample_at_time() [OPTIONAL]", str(e))
    else:
        results.add_warning("Geometry.sample_at_time()", "Method not implemented (optional)")
    
    if verbose:
        results.print_summary()
    
    return results


# ============================================================================
# Test 2: PINN Class
# ============================================================================

def test_pinn(pinn, verbose=True):
    """
    Test the PINN class implementation.
    
    Parameters:
    -----------
    pinn : function
        The student's Pinn model building function
    verbose : bool
        Print detailed test results
    
    Returns:
    --------
    results : TestResults
        Test results object
    """
    results = TestResults()
    
    if verbose:
        print("="*70)
        print("Testing PINN Class")
        print("="*70 + "\n")
    
    inp_sahpe=(2,)
    out_shape=1
    layers=[10,20,10]
    Pinn=pinn(inp_sahpe,out_shape,layers)
    # Test 2.1: Initialization
    try:
        assert hasattr(Pinn, 'layers'), "PINN must have attribute 'layers'"
        assert hasattr(Pinn, 'weights'), "PINN must have attribute 'weights'"
        assert len(Pinn.weights) == (len(Pinn.layers) - 1)*2, \
            f"Expected {(len(Pinn.layers)-1)*2} layer parameters, got {len(Pinn.weights)}"
        
        
        results.add_pass("PINN initialization")
    except Exception as e:
        results.add_fail("PINN initialization", str(e))

    # Test 2.2: Forward pass
    try:
        # Test with a few points
        x_test = np.array([[1.0, 0.5], [2.0, 0.3], [3.0, 0.7]])
        output = Pinn(x_test)
        
        assert output.shape == (3, 1), f"Expected output shape (3, 1), got {output.shape}"
        assert not np.any(np.isnan(output)), "Output contains NaN values"
        assert not np.any(np.isinf(output)), "Output contains Inf values"
        results.add_pass("PINN forward pass")
    except Exception as e:
        results.add_fail("PINN forward pass", str(e))
    
    
    # Test 2.3: Different inputs give different outputs
    try:
        x1 = np.array([[1.0, 0.5]])
        x2 = np.array([[2.0, 0.5]])
        
        out1 = Pinn(x1)
        out2 = Pinn(x2)
        
        # They shouldn't be exactly equal (very unlikely with random init)
        if np.allclose(out1, out2):
            results.add_warning("PINN output variation", 
                              "Different inputs produce very similar outputs")
        else:
            results.add_pass("PINN output variation")
    except Exception as e:
        results.add_fail("PINN output variation", str(e))
    
    
    # Test 2.4: Parameter initialization (check for Glorot/Xavier)
    try:
        # Check that weights are not all zeros
        first_layer_weights = Pinn.weights[0]
        
        assert not np.allclose(first_layer_weights, 0.0), \
            "Weights should not all be zero"
        
        # Check weights are in reasonable range for Glorot initialization
        # Glorot uniform: [-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out))]
        layers=Pinn.layers
        fan_in, fan_out = layers[0].output_shape[0][1:][0], layers[1].output_shape[1:][0]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        
        if np.all(np.abs(first_layer_weights) <= limit * 1.1):  # Allow 10% margin
            results.add_pass("PINN parameter initialization (Glorot-like)")
        else:
            results.add_warning("PINN parameter initialization", 
                              "Weights may not use Glorot initialization")
    except Exception as e:
        results.add_warning("PINN parameter initialization", str(e))
    
    if verbose:
        results.print_summary()

    return results


# ============================================================================
# Test 3: Loss Functions
# ============================================================================

def test_loss_functions(loss_pde, loss_bc, loss_ic, total_loss, Pinn, Geometry, verbose=True):
    """
    Test the loss function implementations.
    
    Parameters:
    -----------
    loss_pde : function
        Student's loss_pde function
    loss_bc : function
        Student's loss_bc function
    loss_ic : function
        Student's loss_ic function
    total_loss : function
        Student's total_loss function
    Pinn : function
        The student's Pinn model building function
    Geometry : class
        Student's Geometry class
    verbose : bool
        Print detailed test results
    
    Returns:
    --------
    results : TestResults
        Test results object
    """
    results = TestResults()
    
    if verbose:
        print("="*70)
        print("Testing Loss Functions")
        print("="*70 + "\n")

    inp_sahpe=(2,)
    out_shape=1
    layers=[10,20,10]
    pinn=Pinn(inp_sahpe,out_shape,layers)

    
    # Setup
    try:
        L = 10.0
        D = 3.0
        geo = Geometry(L=L, t_start=0.0, t_end=1.0)
        
        points_interior = geo.sample_interior(50)
        points_boundary = geo.sample_boundary(20)
        points_initial = geo.sample_initial(20)
    except Exception as e:
        results.add_fail("Loss function test setup", str(e))
        if verbose:
            results.print_summary()
        return results
    
    # Test 3.1: loss_pde
    try:
        x=tf.constant(points_interior[:,0], tf.float32)
        t=tf.constant(points_interior[:,1], tf.float32)
        D=tf.constant(D, tf.float32)
        l_pde = loss_pde(pinn, x,t, D).numpy()
        
        assert isinstance(l_pde, (np.float32, np.array)), \
            f"loss_pde should return a scalar, got {type(l_pde)}"
        
        if isinstance(l_pde, np.ndarray):
            assert l_pde.shape == () or l_pde.shape == (1,), \
                f"loss_pde should return a scalar, got shape {l_pde.shape}"
        
        assert not np.isnan(l_pde), "loss_pde returned NaN"
        assert not np.isinf(l_pde), "loss_pde returned Inf"
        assert l_pde >= 0, f"Loss should be non-negative, got {l_pde}"
        
        results.add_pass("loss_pde()")
    except Exception as e:
        results.add_fail("loss_pde()", str(e))
    
    # Test 3.2: loss_bc
    try:
        x=tf.constant(points_boundary[:,0], tf.float32)
        t=tf.constant(points_boundary[:,1], tf.float32)        
        l_bc = loss_bc(pinn, x,t).numpy()
        
        assert isinstance(l_bc, (np.float32, np.ndarray)), \
            f"loss_bc should return a scalar, got {type(l_bc)}"
        
        if isinstance(l_bc, np.ndarray):
            assert l_bc.shape == () or l_bc.shape == (1,), \
                f"loss_bc should return a scalar, got shape {l_bc.shape}"
        
        assert not np.isnan(l_bc), "loss_bc returned NaN"
        assert not np.isinf(l_bc), "loss_bc returned Inf"
        assert l_bc >= 0, f"Loss should be non-negative, got {l_bc}"
        
        results.add_pass("loss_bc()")
    except Exception as e:
        results.add_fail("loss_bc()", str(e))
    
    # Test 3.3: loss_ic
    try:
        true_ic=np.sin(np.pi * points_boundary[:, 0:1] / L)
        true_ic=tf.constant(true_ic,tf.float32)
        x=tf.constant(points_boundary[:,0], tf.float32)
        t=tf.constant(points_boundary[:,1], tf.float32)  
        l_ic = loss_ic(pinn, x,t,true_ic).numpy()
        
        assert isinstance(l_ic, (np.float32, np.ndarray)), \
            f"loss_ic should return a scalar, got {type(l_ic)}"
        
        if isinstance(l_ic, np.ndarray):
            assert l_ic.shape == () or l_ic.shape == (1,), \
                f"loss_ic should return a scalar, got shape {l_ic.shape}"
        
        assert not np.isnan(l_ic), "loss_ic returned NaN"
        assert not np.isinf(l_ic), "loss_ic returned Inf"
        assert l_ic >= 0, f"Loss should be non-negative, got {l_ic}"
        
        results.add_pass("loss_ic()")
    except Exception as e:
        results.add_fail("loss_ic()", str(e))
    
    # Test 3.4: total_loss
    try:
        result = total_loss(pinn, points_interior, points_boundary, 
                           points_initial, D, L)
        
        # Should return (total, (l_pde, l_bc, l_ic))
        assert isinstance(result, tuple), "total_loss should return a tuple"
        assert len(result) == 2, "total_loss should return (total, components)"
        
        l_total, components = result
        assert not np.isnan(l_total), "total_loss returned NaN"
        assert not np.isinf(l_total), "total_loss returned Inf"
        assert l_total >= 0, f"Loss should be non-negative, got {l_total}"
        
        # Check components
        assert isinstance(components, tuple), "Loss components should be a tuple"
        assert len(components) == 3, "Should return 3 loss components"
        
        results.add_pass("total_loss()")
    except Exception as e:
        results.add_fail("total_loss()", str(e))
    
    # Test 3.5: Check that total loss is sum of components
    try:
        l_total, (l_pde_comp, l_bc_comp, l_ic_comp) = total_loss(
            pinn, points_interior, points_boundary, points_initial, D, L
        )
        
        computed_total = l_pde_comp.numpy() + l_bc_comp.numpy() + l_ic_comp.numpy()
        
        assert np.isclose(l_total.numpy(), computed_total, rtol=1e-5), \
            f"Total loss ({l_total}) should equal sum of components ({computed_total})"
        
        results.add_pass("total_loss consistency")
    except Exception as e:
        results.add_fail("total_loss consistency", str(e))
    
    if verbose:
        results.print_summary()
    
    return results


# ============================================================================
# Test 4: Training Function (optional, basic checks only)
# ============================================================================

def test_train_function(train_pinn, Pinn, Geometry,verbose=True):
    """
    Test the training function (basic functionality only).
    
    Args:
    -----------
    train_pinn : function
        Student's train_pinn function
    Pinn : function
        The student's Pinn model building function
    Geometry : class
        Student's Geometry class
    verbose : bool
        Print detailed test results
    
    Returns:
    --------
    results : TestResults
        Test results object
    """
    results = TestResults()
    
    if verbose:
        print("="*70)
        print("Testing Training Function (Basic Checks)")
        print("="*70 + "\n")
    
    # Test: Can run for a few epochs without crashing
    try:
        L = 10.0
        D = 3.0
        geo = Geometry(L=L, t_start=0.0, t_end=1.0)
        inp_sahpe=(2,)
        out_shape=1
        layers = [10, 10]  # Smaller network for faster testing
        pinn=Pinn(inp_sahpe,out_shape,layers)
        
        # Run for just a few epochs
        if verbose:
            print("Running training for 10 epochs (this may take a moment)...\n")
        
        pinn, loss_history, loss_components = train_pinn(
            pinn, geo, D, L,
            n_epochs=10,  # Very short training
            N1=50, N2=20, N3=20,
            learning_rate=1e-3
        )
        
        # Check return values
        assert isinstance(loss_history, (list, np.ndarray, np.ndarray)), \
            "loss_history should be a list or array"
        assert isinstance(loss_components, dict), \
            "loss_components should be a dictionary"
        
        assert len(loss_history) == 10, \
            f"Expected 10 loss values, got {len(loss_history)}"
        
        # Check that loss components exist
        expected_keys = {'pde', 'bc', 'ic'}
        assert set(loss_components.keys()) == expected_keys, \
            f"loss_components should have keys {expected_keys}"
        
        results.add_pass("train_pinn() basic functionality")
        
        # Check if loss decreased
        if loss_history[0] > loss_history[-1]:
            results.add_pass("Training reduces loss")
        else:
            results.add_warning("Training loss", 
                              "Loss did not decrease (may need more epochs)")
        
    except Exception as e:
        results.add_fail("train_pinn() basic functionality", str(e))
    
    if verbose:
        results.print_summary()
    
    return results


# ============================================================================
# Master test function
# ============================================================================

def run_all_tests(Geometry, Pinn, loss_pde, loss_bc, loss_ic, total_loss, 
                  train_pinn=None, verbose=True):
    """
    Run all tests and provide a comprehensive report.
    
    Parameters:
    -----------
    Geometry : class
        Student's Geometry class
    Pinn : function
        The student's Pinn model building function
    loss_pde, loss_bc, loss_ic, total_loss : functions
        Student's loss functions
    train_pinn : function, optional
        Student's training function
    verbose : bool
        Print detailed output
    
    Returns:
    --------
    all_passed : bool
        True if all tests passed
    """
    print("\n" + "="*70)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("="*70 + "\n")
    
    all_results = []
    
    # Test 1: Geometry
    all_results.append(test_geometry(Geometry, verbose=verbose))
    
    # Test 2: PINN
    all_results.append(test_pinn(Pinn, verbose=verbose))
    
    # Test 3: Loss functions
    all_results.append(test_loss_functions(loss_pde, loss_bc, loss_ic, total_loss,
                                          Pinn, Geometry, verbose=verbose))
    
    # Test 4: Training (optional)
    if train_pinn is not None:
        all_results.append(test_train_function(train_pinn, Pinn, Geometry, verbose=verbose))
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL TEST SUMMARY")
    print("="*70)
    
    total_passed = sum(len(r.passed) for r in all_results)
    total_failed = sum(len(r.failed) for r in all_results)
    total_warnings = sum(len(r.warnings) for r in all_results)
    
    print(f"\n✅ Total Passed: {total_passed}")
    print(f"❌ Total Failed: {total_failed}")
    print(f"⚠️  Total Warnings: {total_warnings}")
    
    if total_failed == 0:
        print("\n🎉 Congratulations! All tests passed! 🎉")
        print("="*70 + "\n")
        return True
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        print("="*70 + "\n")
        return False
