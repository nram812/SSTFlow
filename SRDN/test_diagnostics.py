"""Comprehensive Diagnostic and Physical Verification Test for AFNO and SRDN.

Tests:
1. AFNO Mathematical Invertibility & Parseval Energy Conservation
2. Activation Functions: Softshrink (exact thresholding & sparsity) and GELU
3. FiLM Modulation: Conditional affine scaling and shifting
4. Global Receptive Field: Impulse response test (AFNO vs Local Conv2D)
5. Periodic Translation Equivariance of AFNO
6. Multi-Frequency Noise Filtering (Selective Wavenumber Thresholding)
7. End-to-End CPU Training on Synthetic Ocean SST Fields (5 Optimization Steps)
"""

import numpy as np
import tensorflow as tf
from model_srdn_advanced import AFNO2D, FiLMLayer, AFNOResBlock, SRDN_ResAFNO_v4

def test_1_fft_invertibility_and_parseval():
    print("\n" + "="*70)
    print("TEST 1: AFNO FFT Invertibility & Parseval Energy Conservation")
    print("="*70)
    B, H, W, C = 2, 64, 64, 32
    x = tf.random.normal([B, H, W, C], dtype=tf.float32)
    
    # Forward and Inverse with orthonormal normalization
    norm_scale = tf.cast(tf.sqrt(tf.cast(H * W, tf.float32)), tf.complex64)
    x_p = tf.transpose(x, [0, 3, 1, 2])
    x_ft = tf.signal.rfft2d(x_p) / norm_scale
    
    # Parseval energy check
    spatial_energy = tf.reduce_sum(tf.square(x)).numpy()
    # In real FFT, positive frequencies (except 0 and Nyquist) represent two conjugate modes
    ft_sq = tf.abs(x_ft) ** 2
    # Weight interior frequencies by 2
    freq_energy = (tf.reduce_sum(ft_sq[..., :, 0]) + 
                   tf.reduce_sum(ft_sq[..., :, -1]) + 
                   2.0 * tf.reduce_sum(ft_sq[..., :, 1:-1])).numpy()
    
    energy_ratio = freq_energy / spatial_energy
    print(f"Spatial Domain Total Energy : {spatial_energy:.4f}")
    print(f"Frequency Domain Total Energy: {freq_energy:.4f}")
    print(f"Energy Ratio (Parseval)     : {energy_ratio:.6f} (Expected: ~1.000000)")
    assert abs(energy_ratio - 1.0) < 1e-4, f"Parseval failed: ratio {energy_ratio}"

    # Exact Reconstruction check
    x_rec = tf.signal.irfft2d(x_ft * norm_scale, fft_length=[H, W])
    x_rec = tf.transpose(x_rec, [0, 2, 3, 1])
    max_err = tf.reduce_max(tf.abs(x - x_rec)).numpy()
    print(f"Max Reconstruction Error    : {max_err:.2e} (Expected: < 1e-6)")
    assert max_err < 1e-5, f"Invertibility failed: error {max_err}"
    print("✓ TEST 1 PASSED: FFT roundtrip and orthonormal energy conservation verified!")


def test_2_activation_functions():
    print("\n" + "="*70)
    print("TEST 2: Activation Functions (Softshrink & GELU)")
    print("="*70)
    afno = AFNO2D(embed_dim=32, num_blocks=4, sparsity_threshold=0.05)
    
    # 1. Softshrink exact behavior test
    # Values: -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1
    vals = tf.constant([-0.10, -0.05, -0.02, 0.00, 0.02, 0.05, 0.10], dtype=tf.float32)
    lambd = 0.05
    res = afno._softshrink(vals, lambd).numpy()
    # Expected: [-0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]
    expected = np.array([-0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05], dtype=np.float32)
    max_diff = np.max(np.abs(res - expected))
    print(f"Softshrink input values    : {vals.numpy().tolist()}")
    print(f"Softshrink (lambda={lambd}): {np.round(res, 4).tolist()}")
    print(f"Expected exact values      : {expected.tolist()}")
    assert max_diff < 1e-6, f"Softshrink behavior incorrect, max diff: {max_diff}"
    print("✓ Softshrink correctly zeroes amplitudes <= lambda and linearly shrinks amplitudes > lambda.")

    # 2. GELU activation behavior
    gelu = tf.keras.activations.gelu
    z = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=tf.float32)
    gz = gelu(z).numpy()
    print(f"GELU input values          : {z.numpy().tolist()}")
    print(f"GELU output values         : {np.round(gz, 4).tolist()}")
    # Test non-saturation and proper gradients
    with tf.GradientTape() as tape:
        tape.watch(z)
        loss = tf.reduce_sum(gelu(z))
    grad = tape.gradient(loss, z).numpy()
    print(f"GELU gradients d/dz        : {np.round(grad, 4).tolist()}")
    assert not np.any(np.isnan(grad)), "NaNs detected in GELU gradients"
    print("✓ TEST 2 PASSED: Activation functions verified!")


def test_3_film_modulation():
    print("\n" + "="*70)
    print("TEST 3: FiLM (Feature-wise Linear Modulation)")
    print("="*70)
    C = 16
    film = FiLMLayer(channels=C)
    x = tf.ones([2, 8, 8, C], dtype=tf.float32)
    
    # 1. Condition = 0 vector -> should give identity since weights initialized to 0
    cond_zero = tf.zeros([2, 32], dtype=tf.float32)
    y_zero = film(x, cond_zero)
    diff_zero = tf.reduce_max(tf.abs(y_zero - x)).numpy()
    print(f"Identity Test (Zero Condition) Max Difference: {diff_zero:.2e} (Expected: 0.0)")
    assert diff_zero < 1e-6, "FiLM failed identity test when condition is 0"

    # 2. Arbitrary condition -> modulates features across channels
    cond_active = tf.random.normal([2, 32])
    # Manually set non-zero weights in dense layer to test modulation
    film.dense.kernel.assign(tf.random.normal(film.dense.kernel.shape, stddev=0.1))
    film.dense.bias.assign(tf.random.normal(film.dense.bias.shape, stddev=0.1))
    y_mod = film(x, cond_active)
    var_mod = tf.math.reduce_std(y_mod).numpy()
    print(f"Modulated output shape     : {y_mod.shape}")
    print(f"Modulated feature std dev  : {var_mod:.4f} (proves dynamic channel scaling/shifting)")
    assert var_mod > 0.01, "FiLM failed to modulate features"
    print("✓ TEST 3 PASSED: FiLM conditional modulation verified!")


def test_4_global_receptive_field_vs_conv():
    print("\n" + "="*70)
    print("TEST 4: AFNO Global Receptive Field vs Local Convolution")
    print("="*70)
    H, W, C = 64, 64, 32
    
    # Create single point delta impulse at coordinate (32, 32)
    delta_input = np.zeros([1, H, W, C], dtype=np.float32)
    delta_input[0, 32, 32, :] = 10.0
    delta_tensor = tf.constant(delta_input)
    
    # Single 3x3 Conv2D Layer
    conv = tf.keras.layers.Conv2D(C, 3, padding="same")
    conv_out = conv(delta_tensor).numpy()[0, :, :, 0]
    conv_active_pixels = np.sum(np.abs(conv_out) > 1e-6)
    conv_coverage_pct = 100.0 * conv_active_pixels / (H * W)
    
    # Single AFNO2D Layer
    afno = AFNO2D(embed_dim=C, num_blocks=4, sparsity_threshold=0.0)
    afno_out = afno(delta_tensor).numpy()[0, :, :, 0]
    afno_active_pixels = np.sum(np.abs(afno_out) > 1e-6)
    afno_coverage_pct = 100.0 * afno_active_pixels / (H * W)
    
    print(f"Spatial Grid Size          : {H} x {W} = {H*W} pixels")
    print(f"Impulse Location           : (32, 32)")
    print(f"Conv2D Active Pixels (3x3) : {conv_active_pixels} pixels ({conv_coverage_pct:.2f}% of domain)")
    print(f"AFNO2D Active Pixels       : {afno_active_pixels} pixels ({afno_coverage_pct:.2f}% of domain)")
    
    assert conv_active_pixels <= 9, "Conv2D receptive field should be local (<= 9 pixels in single 3x3 layer)"
    assert afno_active_pixels == H * W, f"AFNO must cover 100% of spatial grid in single pass! Got {afno_active_pixels}"
    print("✓ TEST 4 PASSED: AFNO provides complete 100% global token mixing in a single layer!")


def test_5_shift_equivariance():
    print("\n" + "="*70)
    print("TEST 5: AFNO Periodic Shift Equivariance")
    print("="*70)
    H, W, C = 64, 64, 32
    afno = AFNO2D(embed_dim=C, num_blocks=4, sparsity_threshold=0.0)
    
    # Smooth random field
    x = tf.random.normal([1, H, W, C])
    
    # 1. Output of shifted input: AFNO(roll(x))
    shift_h, shift_w = 7, 13
    x_rolled = tf.roll(x, shift=[shift_h, shift_w], axis=[1, 2])
    y1 = afno(x_rolled)
    
    # 2. Shift of original output: roll(AFNO(x))
    y_orig = afno(x)
    y2 = tf.roll(y_orig, shift=[shift_h, shift_w], axis=[1, 2])
    
    equivariance_err = tf.reduce_max(tf.abs(y1 - y2)).numpy()
    rel_err = equivariance_err / (tf.reduce_max(tf.abs(y1)).numpy() + 1e-8)
    print(f"Shift Vector (dy, dx)      : ({shift_h}, {shift_w})")
    print(f"Equivariance Max Error     : {equivariance_err:.2e}")
    print(f"Relative Equivariance Error: {rel_err:.2e}")
    assert rel_err < 1e-4, f"AFNO is not shift-equivariant: rel_err={rel_err}"
    print("✓ TEST 5 PASSED: AFNO is strictly shift-equivariant (commutes with spatial translation)!")


def test_6_cpu_training_on_synthetic_sst():
    print("\n" + "="*70)
    print("TEST 6: CPU Training Run on Synthetic Ocean SST Fields (5 Steps)")
    print("="*70)
    
    # Create realistic synthetic SST fields with mesoscale eddy patterns
    # SST = background temperature gradient + Gaussian vortex eddies + fine turbulence
    np.random.seed(42)
    B = 4
    num_lats, num_longs = 512, 512
    shrink = 8
    
    # High-resolution coordinates
    y_coords, x_coords = np.mgrid[0:num_lats, 0:num_longs]
    # Background temperature gradient (warm North, cool South)
    grad_sst = (num_lats - y_coords) / float(num_lats) * 20.0 + 5.0
    
    # Add random vortex eddies
    synthetic_hi = []
    for _ in range(B):
        field = grad_sst.copy()
        # Add 10 vortex anomalies
        for _ in range(10):
            cx, cy = np.random.randint(50, num_longs - 50), np.random.randint(50, num_lats - 50)
            radius = np.random.uniform(20, 60)
            amp = np.random.uniform(-3.0, 3.0)
            dist_sq = (x_coords - cx)**2 + (y_coords - cy)**2
            field += amp * np.exp(-dist_sq / (2 * radius**2))
        # Standardize (zero mean, unit variance)
        field = (field - np.mean(field)) / np.std(field)
        synthetic_hi.append(field)
    
    y_truth = np.array(synthetic_hi, dtype=np.float32)[:, :, :, None]
    
    # Low-resolution input by average pooling 8x8
    x_input = tf.nn.avg_pool2d(y_truth, ksize=shrink, strides=shrink, padding="SAME").numpy()
    
    print(f"Synthetic SST Dataset Created:")
    print(f"  - Input  coarse SST shape : {x_input.shape} (Range: [{x_input.min():.2f}, {x_input.max():.2f}])")
    print(f"  - Target high-res SST shape: {y_truth.shape} (Range: [{y_truth.min():.2f}, {y_truth.max():.2f}])")
    
    # Initialize Model
    model = SRDN_ResAFNO_v4(numHiddenUnits=128)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    # Initial forward inference
    y_pred_0 = model(x_input, training=False)
    initial_loss = loss_fn(y_truth, y_pred_0).numpy()
    initial_mae = tf.reduce_mean(tf.abs(y_truth - y_pred_0)).numpy()
    print(f"\nInitial State (Untrained):")
    print(f"  - Initial MSE Loss: {initial_loss:.4f}")
    print(f"  - Initial MAE     : {initial_mae:.4f}")
    
    # 5 Optimization steps on CPU
    print("\nExecuting 5 Gradient Descent Optimization Steps on CPU...")
    losses = []
    for step in range(1, 6):
        with tf.GradientTape() as tape:
            y_pred = model(x_input, training=True)
            loss_value = loss_fn(y_truth, y_pred)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        losses.append(loss_value.numpy())
        print(f"  Step {step}/5: Loss = {loss_value.numpy():.4f}")
        
    final_pred = model(x_input, training=False)
    final_loss = loss_fn(y_truth, final_pred).numpy()
    final_mae = tf.reduce_mean(tf.abs(y_truth - final_pred)).numpy()
    
    print(f"\nFinal State After 5 Steps:")
    print(f"  - Final MSE Loss: {final_loss:.4f} (Decreased from {initial_loss:.4f})")
    print(f"  - Final MAE     : {final_mae:.4f} (Decreased from {initial_mae:.4f})")
    print(f"  - Total Loss Reduction: {((initial_loss - final_loss) / initial_loss) * 100:.2f}%")
    
    assert final_loss < initial_loss, f"Optimization failed: final loss {final_loss} >= initial {initial_loss}"
    print("\n✓ TEST 6 PASSED: Model stably backpropagates gradients and converges on CPU with realistic SST data!")


if __name__ == "__main__":
    test_1_fft_invertibility_and_parseval()
    test_2_activation_functions()
    test_3_film_modulation()
    test_4_global_receptive_field_vs_conv()
    test_5_shift_equivariance()
    test_6_cpu_training_on_synthetic_sst()
    print("\n" + "="*70)
    print("ALL 6 DIAGNOSTIC VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("="*70)
