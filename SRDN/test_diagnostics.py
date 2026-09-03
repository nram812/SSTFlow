"""CPU diagnostics for the AFNO layer and mask-aware ResAFNO model.

These tests deliberately check properties that follow from the implementation:
Fourier round trips, softshrink, FiLM identity, nonlocal perturbations,
translation equivariance, finite gradients, and exact coarse consistency.
The old assertion that a single AFNO impulse must activate every pixel was
removed: a frequency-shared linear Fourier map can be globally supported while
still producing exact zeros for some channels/pixels.
"""

import numpy as np
import tensorflow as tf

from model_srdn_advanced import (
    AFNO2D,
    CoarseConsistencyProjection,
    FiLMLayer,
    SRDN_ResAFNO_v4,
)


def test_1_fft_invertibility_and_parseval():
    print("\n" + "=" * 70)
    print("TEST 1: AFNO FFT invertibility and Parseval energy conservation")
    B, H, W, C = 2, 64, 64, 32
    x = tf.random.normal([B, H, W, C], dtype=tf.float32)
    norm_scale = tf.cast(tf.sqrt(tf.cast(H * W, tf.float32)), tf.complex64)
    x_p = tf.transpose(x, [0, 3, 1, 2])
    x_ft = tf.signal.rfft2d(x_p) / norm_scale
    spatial_energy = tf.reduce_sum(tf.square(x)).numpy()
    ft_sq = tf.abs(x_ft) ** 2
    freq_energy = (
        tf.reduce_sum(ft_sq[..., :, 0])
        + tf.reduce_sum(ft_sq[..., :, -1])
        + 2.0 * tf.reduce_sum(ft_sq[..., :, 1:-1])
    ).numpy()
    energy_ratio = freq_energy / spatial_energy
    x_rec = tf.signal.irfft2d(x_ft * norm_scale, fft_length=[H, W])
    x_rec = tf.transpose(x_rec, [0, 2, 3, 1])
    max_err = tf.reduce_max(tf.abs(x - x_rec)).numpy()
    print(f"Parseval ratio={energy_ratio:.7f}; max round-trip error={max_err:.2e}")
    assert abs(energy_ratio - 1.0) < 1e-4
    assert max_err < 1e-5


def test_2_activation_functions():
    print("\nTEST 2: softshrink and GELU")
    afno = AFNO2D(embed_dim=32, num_blocks=4, sparsity_threshold=0.05)
    values = tf.constant([-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10])
    result = afno._softshrink(values, 0.05).numpy()
    expected = np.array([-0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05], np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-6)
    z = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0])
    with tf.GradientTape() as tape:
        tape.watch(z)
        loss = tf.reduce_sum(tf.keras.activations.gelu(z))
    gradient = tape.gradient(loss, z).numpy()
    assert np.isfinite(gradient).all()


def test_3_film_modulation():
    print("\nTEST 3: FiLM identity and modulation")
    channels = 16
    film = FiLMLayer(channels=channels)
    x = tf.ones([2, 8, 8, channels])
    np.testing.assert_allclose(film(x, tf.zeros([2, 32])).numpy(), x.numpy(), atol=1e-6)
    film.dense.kernel.assign(tf.random.normal(film.dense.kernel.shape, stddev=0.1))
    film.dense.bias.assign(tf.random.normal(film.dense.bias.shape, stddev=0.1))
    assert tf.math.reduce_std(film(x, tf.random.normal([2, 32]))).numpy() > 0.01


def test_4_controlled_nonlocal_perturbation():
    """AFNO responds at a distant pixel; this is the valid globality test."""
    print("\nTEST 4: controlled nonlocal perturbation versus local convolution")
    H, W, C = 64, 64, 32
    tf.random.set_seed(42)
    base = tf.random.normal([1, H, W, C])
    changed = base.numpy().copy()
    changed[0, 4, 7, :] += 1.0
    changed = tf.constant(changed)

    conv = tf.keras.layers.Conv2D(C, 3, padding="same")
    conv_delta = np.abs(conv(changed).numpy() - conv(base).numpy())[0, 32, 32, 0]
    afno = AFNO2D(embed_dim=C, num_blocks=4, sparsity_threshold=0.01)
    afno_delta = np.abs(afno(changed).numpy() - afno(base).numpy())[0, 32, 32, 0]
    print(f"distant conv response={conv_delta:.3e}; AFNO response={afno_delta:.3e}")
    assert conv_delta < 1e-7
    assert afno_delta > 1e-8


def test_5_shift_equivariance():
    print("\nTEST 5: periodic translation equivariance")
    H, W, C = 64, 64, 32
    afno = AFNO2D(embed_dim=C, num_blocks=4, sparsity_threshold=0.0)
    x = tf.random.normal([1, H, W, C])
    shift = [7, 13]
    shifted = afno(tf.roll(x, shift=shift, axis=[1, 2]))
    expected = tf.roll(afno(x), shift=shift, axis=[1, 2])
    relative = tf.reduce_max(tf.abs(shifted - expected)).numpy() / (
        tf.reduce_max(tf.abs(shifted)).numpy() + 1e-8
    )
    print(f"relative equivariance error={relative:.3e}")
    assert relative < 1e-4


def test_6_cpu_training_on_synthetic_sst():
    print("\nTEST 6: five-step CPU training on synthetic 16x SST fields")
    np.random.seed(42)
    tf.random.set_seed(42)
    B, fine, shrink = 2, 512, 16
    y_grid, x_grid = np.mgrid[0:fine, 0:fine]
    truth = []
    for _ in range(B):
        field = (fine - y_grid) / float(fine) * 2.0
        for _ in range(6):
            cx, cy = np.random.randint(40, fine - 40, size=2)
            radius = np.random.uniform(20, 50)
            amplitude = np.random.uniform(-0.5, 0.5)
            field += amplitude * np.exp(
                -((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * radius**2)
            )
        truth.append((field - np.mean(field)) / np.std(field))
    truth = np.asarray(truth, dtype=np.float32)[..., None]
    coarse = tf.nn.avg_pool2d(truth, ksize=shrink, strides=shrink, padding="VALID")
    inputs = {
        "coarse_sst": coarse,
        "coarse_mask": tf.ones_like(coarse),
        "fine_mask": tf.ones_like(truth),
    }
    model = SRDN_ResAFNO_v4(
        numHiddenUnits=32,
        numLats=fine,
        numLongs=fine,
        shrink=shrink,
        trunk_blocks=2,
        num_freq_blocks=4,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def loss_value():
        return tf.reduce_mean(tf.square(model(inputs, training=False) - truth))

    initial = float(loss_value().numpy())
    for _ in range(5):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model(inputs, training=True) - truth))
        gradients = tape.gradient(loss, model.trainable_variables)
        assert all(gradient is not None for gradient in gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    final = float(loss_value().numpy())
    print(f"initial loss={initial:.6f}; final loss={final:.6f}")
    assert np.isfinite(final)
    assert final < initial


def test_7_coarse_projection_and_land_mask():
    print("\nTEST 7: exact coarse consistency and hard land mask")
    projection = CoarseConsistencyProjection(shrink=2)
    values = tf.zeros([1, 4, 4, 1])
    coarse = tf.constant([[[[2.0], [3.0]], [[4.0], [5.0]]]])
    coarse_mask = tf.ones_like(coarse)
    fine_mask = np.ones([1, 4, 4, 1], dtype=np.float32)
    fine_mask[:, :2, :2, :] = 0.0
    output = projection([values, coarse, coarse_mask, fine_mask]).numpy()
    assert np.all(output[fine_mask == 0.0] == 0.0)
    for row, col in ((0, 1), (1, 0), (1, 1)):
        block = output[0, row * 2 : row * 2 + 2, col * 2 : col * 2 + 2, 0]
        np.testing.assert_allclose(block.mean(), coarse.numpy()[0, row, col, 0], atol=1e-6)


if __name__ == "__main__":
    for test in (
        test_1_fft_invertibility_and_parseval,
        test_2_activation_functions,
        test_3_film_modulation,
        test_4_controlled_nonlocal_perturbation,
        test_5_shift_equivariance,
        test_6_cpu_training_on_synthetic_sst,
        test_7_coarse_projection_and_land_mask,
    ):
        test()
    print("\nALL AFNO/SRDN DIAGNOSTICS PASSED")
